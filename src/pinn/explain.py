"""Explainability module for the FourierPINN thermal surrogate."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.geometry import Geometry
from .data_loader import NormStats, ScenarioData, power_at_colloc_points
from .evaluate import predict_scenario
from .model import FourierPINN
from .physics import pde_residual, thermal_conductivity

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper — mirrors _build_layer_tensors from trainer.py (2 lines)
# ---------------------------------------------------------------------------

def _layer_tensors(
    geometry: Geometry, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (layer_k, si_layer_mask) tensors for physics computations."""
    layer_k = torch.tensor(
        [l.k_thermal for l in geometry.layers], dtype=torch.float32, device=device
    )
    si_mask = torch.tensor(
        [l.material == 'silicon' for l in geometry.layers], dtype=torch.bool, device=device
    )
    return layer_k, si_mask


# ---------------------------------------------------------------------------
# Option 1 — PDE residual map
# ---------------------------------------------------------------------------

def residual_map(
    model: FourierPINN,
    sc: ScenarioData,
    geometry: Geometry,
    norm_stats: NormStats,
    device: torch.device,
) -> np.ndarray:
    """Compute the absolute PDE residual |∇·(k∇T) + Q| at every data-grid point."""
    model.eval()
    layer_k, si_mask = _layer_tensors(geometry, device)

    htc_t  = torch.tensor(sc.htc_norm,   dtype=torch.float32, device=device)
    tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=device)
    tsv_t  = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=device)

    # Assign actual power density to each grid point
    col_power = power_at_colloc_points(
        sc.coords, sc.layer_ids,
        geometry, sc.power_blocks_wcm2 or {},
        sc.geom_extents,
        norm_stats.power_mean, norm_stats.power_std,
        device,
    )

    res = pde_residual(
        model,
        sc.coords.clone(),      # pde_residual sets requires_grad internally
        sc.layer_ids,
        col_power,
        htc_t, tamb_t, tsv_t,
        layer_k, si_mask,
        norm_stats.T_min, norm_stats.T_max,
        tuple(sc.geom_extents),
        norm_stats.power_std,
    )

    return res.detach().cpu().numpy().__abs__()


# ---------------------------------------------------------------------------
# Option 2a — Power block sensitivity (thermal influence coefficients)
# ---------------------------------------------------------------------------

def _forward_K(
    model: FourierPINN,
    sc: ScenarioData,
    norm_stats: NormStats,
    device: torch.device,
    power_override: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Run a single forward pass and return temperature in Kelvin (N,)."""
    model.eval()
    T_range = norm_stats.T_max - norm_stats.T_min

    power = power_override if power_override is not None else sc.power
    htc_t  = torch.tensor(sc.htc_norm,   dtype=torch.float32, device=device)
    tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=device)
    tsv_t  = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=device)

    with torch.no_grad():
        T_hat = model(sc.coords, sc.layer_ids, power, htc_t, tamb_t, tsv_t)

    return (T_hat.cpu().numpy() * T_range + norm_stats.T_min).astype(np.float32)


def power_block_sensitivity(
    model: FourierPINN,
    sc: ScenarioData,
    geometry: Geometry,
    norm_stats: NormStats,
    device: torch.device,
    delta_wcm2: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Thermal influence coefficients: dT[i]/dQ_k [K per W/cm²] for each block k."""
    layer_name_to_idx = {l.name: i for i, l in enumerate(geometry.layers)}

    # Precompute base power field (normalised) from current scenario power_blocks
    base_blocks = dict(sc.power_blocks_wcm2 or {})

    def _power_tensor(blocks_wcm2: Dict[str, float]) -> torch.Tensor:
        """Rebuild normalised power field for a modified power_blocks dict."""
        pw = power_at_colloc_points(
            sc.coords, sc.layer_ids,
            geometry, blocks_wcm2,
            sc.geom_extents,
            norm_stats.power_mean, norm_stats.power_std,
            device,
        )
        return pw

    # Base temperature (used if delta is small enough that base pass suffices)
    T_base = _forward_K(model, sc, norm_stats, device)

    results: Dict[str, np.ndarray] = {}
    for block in geometry.power_blocks:
        bname = block.name
        if bname not in base_blocks:
            _log.debug("Block %s not in scenario power map — skipping", bname)
            continue

        # Perturbed power map: only this block's power changes
        blocks_plus  = {**base_blocks, bname: base_blocks[bname] + delta_wcm2}
        blocks_minus = {**base_blocks, bname: base_blocks[bname] - delta_wcm2}

        T_plus  = _forward_K(model, sc, norm_stats, device, _power_tensor(blocks_plus))
        T_minus = _forward_K(model, sc, norm_stats, device, _power_tensor(blocks_minus))

        results[bname] = (T_plus - T_minus) / (2.0 * delta_wcm2)

    return results


# ---------------------------------------------------------------------------
# Option 2b — HTC sensitivity map (cooling effectiveness)
# ---------------------------------------------------------------------------

def htc_sensitivity_map(
    model: FourierPINN,
    sc: ScenarioData,
    norm_stats: NormStats,
    device: torch.device,
    delta_htc: float = 50.0,
) -> np.ndarray:
    """Cooling effectiveness: dT[i]/dHTC [K per W/m²·K] at each grid point."""
    T_range = norm_stats.T_max - norm_stats.T_min
    htc_range = norm_stats.htc_max - norm_stats.htc_min
    delta_norm = delta_htc / htc_range

    htc_plus  = torch.tensor(sc.htc_norm + delta_norm, dtype=torch.float32, device=device)
    htc_minus = torch.tensor(sc.htc_norm - delta_norm, dtype=torch.float32, device=device)
    tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=device)
    tsv_t  = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        T_plus_norm  = model(sc.coords, sc.layer_ids, sc.power, htc_plus,  tamb_t, tsv_t)
        T_minus_norm = model(sc.coords, sc.layer_ids, sc.power, htc_minus, tamb_t, tsv_t)

    T_plus_K  = T_plus_norm.cpu().numpy()  * T_range + norm_stats.T_min
    T_minus_K = T_minus_norm.cpu().numpy() * T_range + norm_stats.T_min

    return ((T_plus_K - T_minus_K) / (2.0 * delta_htc)).astype(np.float32)


# ---------------------------------------------------------------------------
# Option 5 — MC Dropout uncertainty
# ---------------------------------------------------------------------------

def mc_dropout_uncertainty(
    model: FourierPINN,
    sc: ScenarioData,
    norm_stats: NormStats,
    device: torch.device,
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predictive mean and std via MC Dropout."""
    T_range = norm_stats.T_max - norm_stats.T_min

    htc_t  = torch.tensor(sc.htc_norm,   dtype=torch.float32, device=device)
    tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=device)
    tsv_t  = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=device)

    was_training = model.training
    model.train()  # activate dropout

    samples: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for _ in range(n_samples):
                T_hat = model(sc.coords, sc.layer_ids, sc.power, htc_t, tamb_t, tsv_t)
                samples.append((T_hat.cpu().numpy() * T_range + norm_stats.T_min).astype(np.float32))
    finally:
        model.train(was_training)  # restore original mode

    stack = np.stack(samples, axis=0)  # (n_samples, N)
    return stack.mean(axis=0), stack.std(axis=0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_residual_map(
    residuals: np.ndarray,
    coords_norm: np.ndarray,
    geom_extents: List[float],
    scenario_name: str,
    output_path: Path,
    z_fraction: float = 0.95,
) -> None:
    """Scatter plot of absolute PDE residual at a z-slice."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib not available; skipping plot_residual_map")
        return

    L_x, L_y, L_z = geom_extents
    coords_phys = coords_norm * np.array([L_x, L_y, L_z])

    z_target = z_fraction * L_z
    tol = 0.03 * L_z
    mask = np.abs(coords_phys[:, 2] - z_target) < tol
    if mask.sum() < 10:
        _log.warning("plot_residual_map: too few points near z=%.0f µm", z_target)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sc_plot = ax.scatter(
        coords_phys[mask, 0] / 1000, coords_phys[mask, 1] / 1000,
        c=residuals[mask], cmap='hot_r', s=2,
    )
    plt.colorbar(sc_plot, ax=ax, label='|PDE residual| (normalised)')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'PDE residual map — {scenario_name}\nz ≈ {z_target/1000:.2f} mm')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Option 3 — Targeted Integrated Gradients
# ---------------------------------------------------------------------------

# Ordered names of the 7 differentiable input features.
# layer_id is an integer embedding and excluded from IG (not differentiable).
_IG_FEATURE_NAMES = ['x_pos', 'y_pos', 'z_pos', 'power', 'htc', 't_amb', 'tsv_frac']


def _ig_point(
    model: FourierPINN,
    idx: int,
    sc: ScenarioData,
    norm_stats: NormStats,
    device: torch.device,
    bl_coord: torch.Tensor,
    bl_power: torch.Tensor,
    bl_htc:   torch.Tensor,
    bl_tamb:  torch.Tensor,
    bl_tsv:   torch.Tensor,
    n_steps: int,
) -> Tuple[Dict[str, float], float]:
    """Integrated Gradients for a single grid point."""
    T_range = norm_stats.T_max - norm_stats.T_min

    x_pt  = sc.coords[idx:idx+1].clone().to(device).float()
    p_pt  = sc.power[idx:idx+1].clone().to(device).float()
    lid   = sc.layer_ids[idx:idx+1].to(device)
    h_pt  = torch.tensor([sc.htc_norm],   dtype=torch.float32, device=device)
    ta_pt = torch.tensor([sc.t_amb_norm], dtype=torch.float32, device=device)
    tv_pt = torch.tensor([sc.tsv_frac],   dtype=torch.float32, device=device)

    bx  = bl_coord.to(device).float()
    bp  = bl_power.to(device).float()
    bh  = bl_htc.to(device).float()
    bta = bl_tamb.to(device).float()
    btv = bl_tsv.to(device).float()

    model.eval()
    with torch.no_grad():
        T_bl = model(bx, lid, bp, bh, bta, btv)
        T_baseline_K = float(T_bl.item()) * T_range + norm_stats.T_min

    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)[1:]
    acc = {k: torch.zeros(1, device=device) for k in ('power', 'htc', 'tamb', 'tsv')}
    acc['coord'] = torch.zeros_like(x_pt)

    for alpha in alphas:
        ci  = (bx  + alpha * (x_pt - bx )).detach().requires_grad_(True)
        pi  = (bp  + alpha * (p_pt - bp )).detach().requires_grad_(True)
        hi  = (bh  + alpha * (h_pt - bh )).detach().requires_grad_(True)
        tai = (bta + alpha * (ta_pt - bta)).detach().requires_grad_(True)
        tvi = (btv + alpha * (tv_pt - btv)).detach().requires_grad_(True)

        T = model(ci, lid, pi, hi, tai, tvi)
        g = torch.autograd.grad(T, [ci, pi, hi, tai, tvi], retain_graph=False)
        acc['coord'] += g[0].detach()
        acc['power'] += g[1].detach().view(1)
        acc['htc']   += g[2].detach().view(1)
        acc['tamb']  += g[3].detach().view(1)
        acc['tsv']   += g[4].detach().view(1)

    ig_coord = (acc['coord'] / n_steps * (x_pt - bx)).squeeze(0).cpu().numpy() * T_range
    ig_power = float((acc['power'] / n_steps * (p_pt - bp)).item()) * T_range
    ig_htc   = float((acc['htc']   / n_steps * (h_pt - bh)).item()) * T_range
    ig_tamb  = float((acc['tamb']  / n_steps * (ta_pt - bta)).item()) * T_range
    ig_tsv   = float((acc['tsv']   / n_steps * (tv_pt - btv)).item()) * T_range

    return {
        'x_pos': float(ig_coord[0]), 'y_pos': float(ig_coord[1]),
        'z_pos': float(ig_coord[2]), 'power': ig_power,
        'htc':   ig_htc, 't_amb': ig_tamb, 'tsv_frac': ig_tsv,
    }, T_baseline_K


def hotspot_ig(
    model: FourierPINN,
    sc: ScenarioData,
    norm_stats: NormStats,
    device: torch.device,
    n_steps: int = 50,
    include_z_profile: bool = True,
) -> Dict:
    """Integrated Gradients attribution at the predicted hotspot + z-profile."""
    T_range = norm_stats.T_max - norm_stats.T_min

    bl_coord = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    bl_power = torch.tensor([0.0],             dtype=torch.float32)
    bl_htc   = torch.tensor([0.5],             dtype=torch.float32)
    bl_tamb  = torch.tensor([0.5],             dtype=torch.float32)
    bl_tsv   = torch.tensor([0.0],             dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        htc_t  = torch.tensor(sc.htc_norm,   dtype=torch.float32, device=device)
        tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=device)
        tsv_t  = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=device)
        T_all  = model(sc.coords, sc.layer_ids, sc.power, htc_t, tamb_t, tsv_t)

    hs_idx = int(T_all.argmax().cpu())
    hs_T_K = float(T_all[hs_idx].cpu()) * T_range + norm_stats.T_min

    hs_attrs, baseline_T_K = _ig_point(
        model, hs_idx, sc, norm_stats, device,
        bl_coord, bl_power, bl_htc, bl_tamb, bl_tsv, n_steps,
    )
    completeness_error = abs(sum(hs_attrs.values()) - (hs_T_K - baseline_T_K)) / T_range

    z_profile: List[Dict] = []
    if include_z_profile:
        coords_np    = sc.coords.cpu().numpy()
        layer_ids_np = sc.layer_ids.cpu().numpy()
        T_all_np     = T_all.cpu().numpy()
        hs_xy        = coords_np[hs_idx, :2]

        for layer_idx in range(int(layer_ids_np.max()) + 1):
            mask = layer_ids_np == layer_idx
            if not mask.any():
                continue
            closest = int(np.where(mask)[0][
                ((coords_np[mask, :2] - hs_xy) ** 2).sum(1).argmin()
            ])
            pt_attrs, _ = _ig_point(
                model, closest, sc, norm_stats, device,
                bl_coord, bl_power, bl_htc, bl_tamb, bl_tsv, n_steps,
            )
            z_profile.append({
                'layer_idx': layer_idx,
                'z_um':      float(coords_np[closest, 2]) * sc.geom_extents[2],
                'T_K':       float(T_all_np[closest]) * T_range + norm_stats.T_min,
                'attrs':     pt_attrs,
            })

    return {
        'hotspot_idx':        hs_idx,
        'hotspot_T_K':        hs_T_K,
        'baseline_T_K':       baseline_T_K,
        'completeness_error': completeness_error,
        'hotspot_attrs':      hs_attrs,
        'z_profile':          z_profile,
        'feature_names':      _IG_FEATURE_NAMES,
    }


def plot_ig_attribution(
    ig_result: Dict,
    scenario_name: str,
    output_path: Path,
    geometry=None,
) -> None:
    """
    Two-panel figure: hotspot attribution bar chart (left) + z-profile attribution heatmap (right, feature × layer).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib not available; skipping plot_ig_attribution")
        return

    hs_attrs  = ig_result['hotspot_attrs']
    z_profile = ig_result['z_profile']
    features  = ig_result['feature_names']
    hs_T_K    = ig_result['hotspot_T_K']
    bl_T_K    = ig_result['baseline_T_K']
    compl_err = ig_result['completeness_error']

    feat_labels = ['x', 'y', 'z', 'Q', 'HTC', 'T_amb', 'TSV']
    vals   = [hs_attrs[f] for f in features]
    colors = ['#d62728' if v >= 0 else '#1f77b4' for v in vals]

    n_cols = 2 if z_profile else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    ax = axes[0]
    ax.bar(feat_labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Attribution (K)')
    ax.set_title(
        f'IG attribution at hotspot — {scenario_name}\n'
        f'T_pred={hs_T_K:.1f} K   T_base={bl_T_K:.1f} K   '
        f'err={compl_err:.3f}'
    )
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.02, f'Σ = {sum(vals):+.1f} K', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='gray')

    if z_profile:
        ax2 = axes[1]
        mat  = np.array([[pt['attrs'][f] for f in features] for pt in z_profile])
        vmax = np.abs(mat).max() or 1.0
        im   = ax2.imshow(mat.T, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, origin='lower')
        plt.colorbar(im, ax=ax2, label='Attribution (K)')
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(feat_labels)
        ax2.set_xlabel('Layer (bottom → top)')

        if geometry is not None:
            xlabels = [geometry.layers[pt['layer_idx']].name for pt in z_profile]
        else:
            xlabels = [f"{pt['z_um']/1000:.1f}mm" for pt in z_profile]
        ax2.set_xticks(range(len(z_profile)))
        ax2.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
        ax2.set_title(f'Attribution z-profile — {scenario_name}')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_map(
    sensitivity: np.ndarray,
    coords_norm: np.ndarray,
    geom_extents: List[float],
    title: str,
    output_path: Path,
    z_fraction: float = 0.95,
    cmap: str = 'RdBu_r',
    clabel: str = 'dT/dX',
) -> None:
    """Scatter plot of a sensitivity field at a z-slice."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib not available; skipping plot_sensitivity_map")
        return

    L_x, L_y, L_z = geom_extents
    coords_phys = coords_norm * np.array([L_x, L_y, L_z])

    z_target = z_fraction * L_z
    tol = 0.03 * L_z
    mask = np.abs(coords_phys[:, 2] - z_target) < tol
    if mask.sum() < 10:
        _log.warning("plot_sensitivity_map: too few points near z=%.0f µm", z_target)
        return

    vals = sensitivity[mask]
    vmax = np.abs(vals).max()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc_plot = ax.scatter(
        coords_phys[mask, 0] / 1000, coords_phys[mask, 1] / 1000,
        c=vals, cmap=cmap, s=2, vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(sc_plot, ax=ax, label=clabel)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'{title}\nz ≈ {z_target/1000:.2f} mm')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_uncertainty_map(
    T_std_K: np.ndarray,
    coords_norm: np.ndarray,
    geom_extents: List[float],
    scenario_name: str,
    output_path: Path,
    z_fraction: float = 0.95,
) -> None:
    """Scatter plot of MC Dropout predictive std (K) at a z-slice."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib not available; skipping plot_uncertainty_map")
        return

    L_x, L_y, L_z = geom_extents
    coords_phys = coords_norm * np.array([L_x, L_y, L_z])

    z_target = z_fraction * L_z
    tol = 0.03 * L_z
    mask = np.abs(coords_phys[:, 2] - z_target) < tol
    if mask.sum() < 10:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sc_plot = ax.scatter(
        coords_phys[mask, 0] / 1000, coords_phys[mask, 1] / 1000,
        c=T_std_K[mask], cmap='viridis', s=2, vmin=0,
    )
    plt.colorbar(sc_plot, ax=ax, label='Predictive std (K)')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'MC Dropout uncertainty — {scenario_name}\nz ≈ {z_target/1000:.2f} mm')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
