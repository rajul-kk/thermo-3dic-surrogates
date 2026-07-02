"""
PINN evaluation: metrics and comparison plots.

Metrics reported per scenario and aggregated:
  mae_K            mean absolute error in Kelvin
  rmse_K           root mean squared error
  max_err_K        maximum point error
  r2               coefficient of determination
  hotspot_T_err_K  temperature error at the true hotspot location
  hotspot_loc_err  distance between predicted and true hotspot (µm)
  pde_res_rms      RMS of PDE residual at test points (optional)

Plots:
  - Temperature field z-slice comparison: 3D-ICE vs PINN
  - Error map at a z-slice
  - Z-axis temperature profile through hotspot
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.geometry import Geometry
from .data_loader import NormStats, ScenarioData, ThermalDataset
from .model import FourierPINN

_log = logging.getLogger(__name__)


def predict_scenario(
    model: FourierPINN,
    sc: ScenarioData,
    device: torch.device,
    norm_stats: NormStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single scenario.

    Returns:
        T_pred_K: (N,) predicted temperatures in Kelvin
        T_true_K: (N,) ground-truth temperatures in Kelvin
    """
    model.eval()
    T_range = norm_stats.T_max - norm_stats.T_min

    with torch.no_grad():
        htc_t = torch.tensor(sc.htc_norm, dtype=torch.float32, device=device)
        tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=device)
        tsv_t = torch.tensor(sc.tsv_frac, dtype=torch.float32, device=device)

        T_hat = model(sc.coords, sc.layer_ids, sc.power, htc_t, tamb_t, tsv_t)

    T_pred_K = T_hat.cpu().numpy() * T_range + norm_stats.T_min
    T_true_K = sc.temp.cpu().numpy() * T_range + norm_stats.T_min
    return T_pred_K, T_true_K


def compute_metrics(
    T_pred_K: np.ndarray,
    T_true_K: np.ndarray,
    coords_norm: np.ndarray,
    geom_extents: List[float],
) -> Dict[str, float]:
    """Compute all accuracy metrics for one scenario."""
    err = T_pred_K - T_true_K
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    max_err = float(np.abs(err).max())

    ss_res = float(((T_true_K - T_pred_K) ** 2).sum())
    ss_tot = float(((T_true_K - T_true_K.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Hotspot: true and predicted locations
    true_hs_idx = int(T_true_K.argmax())
    pred_hs_idx = int(T_pred_K.argmax())

    hotspot_T_err = float(abs(T_pred_K[true_hs_idx] - T_true_K[true_hs_idx]))

    # Distance between hotspot locations in µm
    L_x, L_y, L_z = geom_extents
    true_hs_pos = coords_norm[true_hs_idx] * np.array([L_x, L_y, L_z])
    pred_hs_pos = coords_norm[pred_hs_idx] * np.array([L_x, L_y, L_z])
    hotspot_loc_err = float(np.linalg.norm(true_hs_pos - pred_hs_pos))

    return {
        'mae_K': mae,
        'rmse_K': rmse,
        'max_err_K': max_err,
        'r2': r2,
        'hotspot_T_err_K': hotspot_T_err,
        'hotspot_loc_err_um': hotspot_loc_err,
    }


def evaluate_dataset(
    model: FourierPINN,
    dataset: ThermalDataset,
    norm_stats: NormStats,
    device: torch.device,
) -> Dict[str, object]:
    """
    Evaluate model on all scenarios in a dataset.

    Returns dict with per-scenario metrics and aggregate statistics.
    """
    results = {}
    all_mae = []
    all_rmse = []
    all_max_err = []
    all_r2 = []

    for sc in dataset.scenarios:
        T_pred_K, T_true_K = predict_scenario(model, sc, device, norm_stats)
        metrics = compute_metrics(
            T_pred_K, T_true_K,
            sc.coords.cpu().numpy(),
            sc.geom_extents,
        )
        results[sc.name] = metrics
        all_mae.append(metrics['mae_K'])
        all_rmse.append(metrics['rmse_K'])
        all_max_err.append(metrics['max_err_K'])
        all_r2.append(metrics['r2'])

    results['_aggregate'] = {
        'mean_mae_K': float(np.mean(all_mae)),
        'max_mae_K': float(np.max(all_mae)),
        'mean_rmse_K': float(np.mean(all_rmse)),
        'mean_max_err_K': float(np.mean(all_max_err)),
        'mean_r2': float(np.mean(all_r2)),
        'n_scenarios': len(dataset),
    }

    _log.info(
        "Evaluation: mean MAE=%.3f K, max MAE=%.3f K, mean R²=%.5f",
        results['_aggregate']['mean_mae_K'],
        results['_aggregate']['max_mae_K'],
        results['_aggregate']['mean_r2'],
    )
    return results


def save_metrics(results: Dict, output_path: Path) -> None:
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def plot_comparison(
    T_pred_K: np.ndarray,
    T_true_K: np.ndarray,
    coords_norm: np.ndarray,
    geom_extents: List[float],
    scenario_name: str,
    output_path: Path,
    z_fraction: float = 0.95,   # z-slice as fraction of total height (near top die)
) -> None:
    """
    Side-by-side z-slice: 3D-ICE ground truth vs PINN prediction + error map.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        _log.warning("matplotlib not available; skipping plot_comparison")
        return

    L_x, L_y, L_z = geom_extents
    coords_phys = coords_norm * np.array([L_x, L_y, L_z])

    # Select points near the chosen z-slice (within 2% of domain height)
    tol = 0.02 * L_z
    z_target = z_fraction * L_z
    mask = np.abs(coords_phys[:, 2] - z_target) < tol
    if mask.sum() < 10:
        _log.warning("plot_comparison: too few points near z=%.0f µm; skipping", z_target)
        return

    x_slice = coords_phys[mask, 0]
    y_slice = coords_phys[mask, 1]
    T_true_slice = T_true_K[mask]
    T_pred_slice = T_pred_K[mask]
    err_slice = np.abs(T_pred_slice - T_true_slice)

    vmin = min(T_true_slice.min(), T_pred_slice.min())
    vmax = max(T_true_slice.max(), T_pred_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, title in [
        (axes[0], T_true_slice, '3D-ICE (Ground Truth)'),
        (axes[1], T_pred_slice, 'PINN Prediction'),
        (axes[2], err_slice, '|Error| (K)'),
    ]:
        if title.startswith('|Error|'):
            sc = ax.scatter(x_slice / 1000, y_slice / 1000, c=vals,
                            cmap='hot', s=2)
        else:
            sc = ax.scatter(x_slice / 1000, y_slice / 1000, c=vals,
                            cmap='inferno', vmin=vmin, vmax=vmax, s=2)
        plt.colorbar(sc, ax=ax, label='K')
        ax.set_title(title)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

    fig.suptitle(f"{scenario_name} | z ≈ {z_target/1000:.2f} mm")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_z_profile(
    T_pred_K: np.ndarray,
    T_true_K: np.ndarray,
    coords_norm: np.ndarray,
    geom_extents: List[float],
    geometry: Geometry,
    scenario_name: str,
    output_path: Path,
) -> None:
    """Z-axis temperature profile through the hotspot column."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    L_x, L_y, L_z = geom_extents
    coords_phys = coords_norm * np.array([L_x, L_y, L_z])

    # Find hotspot xy location from ground truth
    hs_idx = int(T_true_K.argmax())
    hs_x, hs_y = coords_phys[hs_idx, 0], coords_phys[hs_idx, 1]

    # Points within 200µm of hotspot column
    tol = 200.0
    mask = (
        (np.abs(coords_phys[:, 0] - hs_x) < tol) &
        (np.abs(coords_phys[:, 1] - hs_y) < tol)
    )
    if mask.sum() < 5:
        return

    z_vals = coords_phys[mask, 2]
    order = np.argsort(z_vals)
    z_sorted = z_vals[order]
    T_true_z = T_true_K[mask][order]
    T_pred_z = T_pred_K[mask][order]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(z_sorted / 1000, T_true_z, 'b-o', markersize=3, label='3D-ICE')
    ax.plot(z_sorted / 1000, T_pred_z, 'r--s', markersize=3, label='PINN')

    # Layer boundary lines
    for layer in geometry.layers:
        ax.axvline(layer.z_top / 1000, color='gray', linewidth=0.5, linestyle=':')

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(f"{scenario_name} — z-profile at hotspot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
