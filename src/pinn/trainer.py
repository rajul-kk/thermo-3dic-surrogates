"""
PINN training loop with curriculum staging and adaptive loss weighting.

Training stages:
  Stage 1 (epochs   0-1000): data loss only — network learns the temperature distribution
  Stage 2 (epochs 1000-3000): add PDE + BC at fixed low weights
  Stage 3 (epochs 3000-8000): NTK-based adaptive weights updated every 50 epochs

Optimizer: Adam with CosineAnnealingWarmRestarts (T_0=2000)
Gradient clipping: max_norm=1.0 (PDE second derivatives can spike)
Mixed precision: enabled when CUDA is available
"""

import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..core.geometry import Geometry
from .data_loader import (
    NormStats, ScenarioData, ThermalDataset,
    sample_collocation_points, sample_collocation_stratified,
    sample_bc_top_points, sample_bc_faces_grouped,
    power_at_colloc_points,
)
from .losses import LossWeights, data_loss, pde_loss, bc_loss, interface_loss, total_loss
from .model import FourierPINN
from .physics import (
    pde_residual, bc_residual_top, bc_residual_adiabatic, thermal_conductivity,
)
from .sampling import (
    SAMPLING_STRATEGIES, compute_adaptive_weights,
    importance_resample, curriculum_blend_factor, curriculum_enhanced_resample,
)

_log = logging.getLogger(__name__)


def _build_layer_tensors(
    geometry: Geometry,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (layer_k, si_layer_mask) tensors for physics computations."""
    k_vals = torch.tensor(
        [layer.k_thermal for layer in geometry.layers],
        dtype=torch.float32, device=device
    )
    si_mask = torch.tensor(
        [layer.material == 'silicon' for layer in geometry.layers],
        dtype=torch.bool, device=device
    )
    return k_vals, si_mask


class Trainer:
    """
    Trains a FourierPINN for a single geometry.

    Args:
        model:           FourierPINN instance
        geometry:        target Geometry (single geometry per trainer)
        norm_stats:      global normalization statistics
        train_data:      ThermalDataset with training scenarios
        val_data:        ThermalDataset with validation/test scenarios
        output_dir:      directory for checkpoints and logs
        n_col:           number of collocation points for PDE loss per step
        n_bc_top:        number of BC points on top surface per step
        n_bc_side:       number of BC points per side face per step (ignored when hard_adiabatic)
        epochs:          total training epochs
        lr:              initial Adam learning rate
        device:          training device
        hard_adiabatic:  if True, bc_sides loss is skipped (model.hard_adiabatic handles it)
        log_interval:    print loss every N epochs
        val_interval:    run validation every N epochs.
        sampling_strategy: one of 'rar' (default, PDE-residual-proportional —
            original RAR-D behaviour), 'hessian' (curvature/Laplacian-trace
            weighted), 'importance' (softmax-temperature resampling of the
            residual field — simplified proxy for adversarial adaptive
            sampling), 'curriculum' (blends uniform -> residual-weighted
            sampling over training, see src/pinn/sampling.py for citations).
        sampling_temperature: softmax temperature for 'importance'/'curriculum'
            strategies (lower = sharper toward high-signal points).
        curriculum_warmup_frac: fraction of total epochs to stay at pure
            uniform sampling before ramping toward adaptive (curriculum only).
    """

    def __init__(
        self,
        model: FourierPINN,
        geometry: Geometry,
        norm_stats: NormStats,
        train_data: ThermalDataset,
        val_data: ThermalDataset,
        output_dir: Path,
        n_col: int = 20000,
        n_bc_top: int = 2000,
        n_bc_side: int = 1000,
        epochs: int = 8000,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        hard_adiabatic: bool = True,
        log_interval: int = 100,
        val_interval: Optional[int] = None,
        rar_interval: int = 500,
        rar_add_n: int = 1000,
        rar_max_col: int = 30000,
        rar_epsilon: float = 0.2,
        sampling_strategy: str = 'rar',
        sampling_temperature: float = 1.0,
        curriculum_warmup_frac: float = 0.2,
        use_compile: bool = False,
        use_func_pde: bool = False,
    ):
        self.model = model
        self.geometry = geometry
        self.norm_stats = norm_stats
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_col = n_col
        self.n_bc_top = n_bc_top
        self.n_bc_side = n_bc_side
        self.epochs = epochs
        self.hard_adiabatic = hard_adiabatic
        self.log_interval = log_interval
        self.val_interval = val_interval if val_interval is not None else log_interval

        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model.to(self.device)
        train_data.to_device(self.device)
        val_data.to_device(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=2000, T_mult=2, eta_min=1e-6
        )
        self.weights = LossWeights()
        self.layer_k, self.si_mask = _build_layer_tensors(geometry, self.device)

        extents = norm_stats.geom_extents.get(geometry.name, [1.0, 1.0, 1.0])
        self.geom_scale = tuple(extents)         # (L_x, L_y, L_z) µm
        # Convective (HTC) BC applies at layer 0 (z=0) — matches 3D-ICE's
        # "bottom heat sink" directive in ice_simulator.py. Was previously
        # len(geometry.layers)-1 (topmost layer, z=1), which enforced the
        # convective condition at the wrong end of the stack relative to the
        # ground truth for every geometry — see geometry_builders.py notes.
        self.convective_layer_id = 0

        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # ── GPU efficiency upgrades ────────────────────────────────────
        # Upgrade 1: torch.compile — fuses ops, reduces kernel launch overhead.
        # ~2-3× speedup on CUDA.  Disabled on CPU (compile overhead exceeds gain).
        self.use_func_pde = use_func_pde
        if use_compile and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                _log.info("torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                _log.warning("torch.compile failed (%s) — continuing without", e)

        # Upgrade 2: func-pde — use torch.func vmap+grad instead of
        # create_graph=True autograd.  Eliminates retained computation graphs.
        if use_func_pde:
            from .physics_func import _has_func, pde_residual_func as _prf
            if _has_func():
                self._pde_residual_fn = _prf
                _log.info("torch.func PDE residual enabled (no create_graph)")
            else:
                self.use_func_pde = False
                _log.warning("torch.func not available — falling back to autograd PDE")

        self.best_val_mae = float('inf')
        self.history: Dict[str, List[float]] = {
            'train_total': [], 'train_data': [], 'train_pde': [], 'train_bc': [],
            'val_mae_K': [], 'epoch': [],
        }

        # RAR-D adaptive sampling parameters
        self.rar_interval = rar_interval
        self.rar_add_n = rar_add_n
        self.rar_max_col = rar_max_col
        self.rar_epsilon = rar_epsilon

        if sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(
                f"sampling_strategy must be one of {SAMPLING_STRATEGIES}, got {sampling_strategy!r}"
            )
        self.sampling_strategy = sampling_strategy
        self.sampling_temperature = sampling_temperature
        self.curriculum_warmup_frac = curriculum_warmup_frac

        # Persistent stratified collocation set — initialised once, grown by RAR.
        # Shared across all training scenarios (collocation positions are geometry-only).
        sc0 = train_data.scenarios[0]
        self._col_coords, self._col_ids = sample_collocation_stratified(
            n_col, sc0.geom_extents, geometry, self.device
        )
        # Per-point lateral k overrides for 2p5d_stack geometries; None otherwise.
        # Cached and recomputed only when the collocation set grows (RAR update).
        self._col_k_lat, self._col_si_lat = self._compute_lateral_k(
            self._col_coords, self._col_ids
        )
        # Region IDs for chiplet geometries (geometry4/5): 0=underfill, 1=chipA, 2=chipB.
        # None for non-2p5d geometries — model.forward() ignores None when region_emb is None.
        self._col_region_ids = self._compute_region_ids(self._col_coords)
        # Cache region IDs for the data grid too (all scenarios share same coords).
        self._data_region_ids = self._compute_region_ids(sc0.coords)

        # TensorBoard writer (optional — skip gracefully if not installed)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.output_dir / 'tb_logs'))
        except ImportError:
            _log.warning("tensorboard not installed; skipping TensorBoard logging")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Run full training loop. Returns path to best checkpoint."""
        _log.info("Training %s on %s for %d epochs", self.geometry.name, self.device, self.epochs)
        _log.info("Training scenarios: %d | Validation scenarios: %d",
                  len(self.train_data), len(self.val_data))

        t0 = time.time()
        for epoch in range(self.epochs):
            random.shuffle(self.train_data.scenarios)
            self.weights.apply_curriculum(epoch)
            train_losses = self._train_epoch(epoch)
            self.scheduler.step()

            # Adaptive weight update every 50 epochs in Stage 3
            if self.weights.curriculum_stage(epoch) == 3 and epoch % 50 == 0:
                self._update_adaptive_weights(train_losses)

            # RAR-D: append high-residual collocation points after PDE loss activates
            if (self.rar_interval > 0
                    and epoch >= 1000
                    and epoch % self.rar_interval == 0):
                self._rar_update(epoch)

            if epoch % self.val_interval == 0:
                val_mae = self._validate()

                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    self._save_checkpoint(epoch, val_mae, best=True)

                if epoch % self.log_interval == 0:
                    self._log_epoch(epoch, train_losses, val_mae, time.time() - t0)

        self._save_checkpoint(self.epochs - 1, self.best_val_mae, best=False)
        if self._writer:
            self._writer.close()

        best_ckpt = self.output_dir / f"{self.geometry.name}_best.pt"
        _log.info("Training complete. Best val MAE: %.3f K. Checkpoint: %s",
                  self.best_val_mae, best_ckpt)
        return best_ckpt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses: Dict[str, float] = {}

        # ── Upgrade 2 (DISABLED — see note below) ────────────────────────
        # This used to pre-compute all S scenarios' data loss in ONE shared
        # forward_batched_scenarios call (Fourier + layer features computed
        # once instead of S times), intended as a ~15-25% wall-time saving.
        #
        # That shared computation graph is incompatible with this loop's
        # per-scenario training step: the loop calls optimizer.step() after
        # EACH scenario's backward(), which mutates model parameters
        # in-place — but the retained graph for scenario i+1 still
        # references those same (now-mutated) parameter tensors, so
        # PyTorch's autograd correctly raises "one of the variables needed
        # for gradient computation has been modified by an inplace
        # operation" on the second scenario's backward() call. This isn't
        # fixable with retain_graph=True alone (tried — same crash, just
        # later): the batched-forward-then-per-scenario-step pattern is
        # fundamentally unsafe unless gradients are accumulated across ALL
        # scenarios before any optimizer.step() (a bigger change to
        # training semantics — one step per epoch instead of one step per
        # scenario — not made here to avoid altering convergence behaviour).
        #
        # This only stayed hidden previously because it was never exercised
        # with a real multi-scenario training set (every actual geometry
        # has 15-50 scenarios; a single-scenario dataset never reaches the
        # second loop iteration where the crash occurs).
        #
        # Reverted to per-scenario independent forward passes (safe, matches
        # the existing fallback path in _compute_losses when
        # precomputed_data_loss=None) at the cost of losing the claimed
        # speedup.
        for sc in self.train_data.scenarios:
            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type,
                                dtype=torch.float16 if self.use_amp else torch.float32):
                losses = self._compute_losses(sc)
                L = total_loss(losses, self.weights)

            if self.use_amp:
                self.scaler.scale(L).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                L.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        n = len(self.train_data)
        return {k: v / n for k, v in epoch_losses.items()}

    def _batched_data_loss(self) -> List[torch.Tensor]:
        """
        Run all S training scenarios through forward_batched_scenarios in one
        call, returning a list of per-scenario data-loss tensors (with grad).

        Shared Fourier + layer features computed once → significant GPU saving
        when S=15 and N_data is large (60k pts for geometry1).
        """
        scenarios = self.train_data.scenarios
        S = len(scenarios)
        # Stack per-scenario tensors
        all_power = torch.stack([sc.power for sc in scenarios])     # (S, N)
        all_htc   = torch.tensor([sc.htc_norm   for sc in scenarios],
                                  device=self.device, dtype=torch.float32)
        all_tamb  = torch.tensor([sc.t_amb_norm for sc in scenarios],
                                  device=self.device, dtype=torch.float32)
        all_tsv   = torch.tensor([sc.tsv_frac   for sc in scenarios],
                                  device=self.device, dtype=torch.float32)
        # All scenarios share the same coord/id arrays (same geometry)
        coords   = scenarios[0].coords    # (N, 3)
        layer_ids = scenarios[0].layer_ids  # (N,)
        all_T_true = torch.stack([sc.temp for sc in scenarios])     # (S, N)

        all_tim_k = torch.tensor(
            [sc.tim_k_norm for sc in scenarios],
            device=self.device, dtype=torch.float32,
        ) if self.model.tim_k_input else None

        # Single batched forward
        T_pred_all = self.model.forward_batched_scenarios(
            coords, layer_ids, all_power, all_htc, all_tamb, all_tsv,
            region_ids=self._data_region_ids,
            tim_k_norms=all_tim_k,
        )  # (S, N)

        # Per-scenario MSE (keep gradient)
        return [data_loss(T_pred_all[i], all_T_true[i]) for i in range(S)]

    def _compute_losses(
        self,
        sc: ScenarioData,
        precomputed_data_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components for one scenario."""
        htc_t   = torch.tensor(sc.htc_norm,   dtype=torch.float32, device=self.device)
        tamb_t  = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=self.device)
        tsv_t   = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=self.device)
        tim_k_t = torch.tensor(sc.tim_k_norm, dtype=torch.float32, device=self.device)

        # Data loss: use precomputed batched result when available (Upgrade 2),
        # otherwise fall back to individual forward pass.
        if precomputed_data_loss is not None:
            losses = {'data': precomputed_data_loss}
        else:
            T_pred = self.model(
                sc.coords, sc.layer_ids, sc.power, htc_t, tamb_t, tsv_t,
                region_ids=self._data_region_ids, tim_k_norm=tim_k_t,
            )
            losses = {'data': data_loss(T_pred, sc.temp)}

        stage = self.weights.curriculum_stage
        if callable(stage):
            # stage is a method on LossWeights; call with a dummy epoch to check
            # We check the current weight values instead
            pass

        if self.weights.pde > 0:
            col_coords, col_ids = self._col_coords, self._col_ids
            col_power = power_at_colloc_points(
                col_coords, col_ids,
                self.geometry, sc.power_blocks_wcm2 or {},
                sc.geom_extents,
                self.norm_stats.power_mean, self.norm_stats.power_std,
                self.device,
            )
            power_scale = self.norm_stats.power_std

            # func_pde vmap path does not support region_ids yet — fall back
            use_func = self.use_func_pde and self._col_region_ids is None
            if use_func:
                res = self._pde_residual_fn(
                    self.model, col_coords, col_ids, col_power,
                    htc_t, tamb_t, tsv_t,
                    self.layer_k, self.si_mask,
                    self.norm_stats.T_min, self.norm_stats.T_max,
                    self.geom_scale, power_scale,
                    col_k_lateral=self._col_k_lat,
                    col_si_lateral=self._col_si_lat,
                )
            else:
                res = pde_residual(
                    self.model, col_coords, col_ids, col_power,
                    htc_t, tamb_t, tsv_t,
                    self.layer_k, self.si_mask,
                    self.norm_stats.T_min, self.norm_stats.T_max,
                    self.geom_scale, power_scale,
                    col_k_lateral=self._col_k_lat,
                    col_si_lateral=self._col_si_lat,
                    region_ids_col=self._col_region_ids,
                    tim_k_norm=tim_k_t,
                )
            losses['pde'] = pde_loss(res)

        if self.weights.bc_top > 0:
            # Convective BC at z=0 (layer 0 / heat_sink) — matches 3D-ICE
            # ground truth ("bottom heat sink" directive). outward_normal_sign
            # =-1.0 because the outward normal at z=0 points in -z.
            bc_coords, bc_ids = sample_bc_top_points(
                self.n_bc_top, self.device, self.convective_layer_id, z_value=0.0
            )
            bc_region_ids = self._compute_region_ids(bc_coords)
            res_top = bc_residual_top(
                self.model, bc_coords, bc_ids,
                torch.zeros(self.n_bc_top, device=self.device),
                htc_t, tamb_t, tsv_t,
                self.layer_k, self.si_mask,
                self.norm_stats.T_min, self.norm_stats.T_max,
                self.geom_scale[2],   # L_z
                sc.htc, sc.t_amb_K,
                region_ids_top=bc_region_ids,
                tim_k_norm=tim_k_t,
                outward_normal_sign=-1.0,
            )
            losses['bc_top'] = bc_loss(res_top)

        # bc_sides: lateral (x,y) walls skipped when hard_adiabatic=True — the
        # cosine coordinate fold in model._hard_adiabatic_transform enforces
        # dT/dn=0 exactly there at init. The z=1 face (top, near die/TIM2) is
        # NEVER covered by hard_adiabatic (the cosine fold only touches x,y),
        # so it is always soft-enforced here, regardless of hard_adiabatic —
        # it is the correct adiabatic face now that z=0 carries the convective
        # BC instead (see sample_bc_faces_grouped docstring).
        if self.weights.bc_sides > 0:
            face_residuals = []
            if not self.hard_adiabatic:
                face_groups = sample_bc_faces_grouped(
                    self.n_bc_side, self.device, self.geometry, sc.geom_extents
                )
                for face_coords, face_ids, normal_dim in face_groups:
                    res = bc_residual_adiabatic(
                        self.model, face_coords, face_ids,
                        torch.zeros(face_coords.shape[0], device=self.device),
                        htc_t, tamb_t, tsv_t,
                        normal_dim=normal_dim, geom_scale=self.geom_scale,
                    )
                    face_residuals.append(res)
            else:
                # hard_adiabatic=True: only the z=1 top face needs a soft term
                # (x,y already exact via the cosine fold).
                top_coords = torch.rand(self.n_bc_side, 3, device=self.device)
                top_coords[:, 2] = 1.0
                top_coords_phys = top_coords.detach().cpu().numpy() * np.array(sc.geom_extents)
                top_ids_np = np.array([
                    max(0, self.geometry.get_layer_index_at_z(float(z)))
                    for z in top_coords_phys[:, 2]
                ], dtype=np.int64)
                top_ids = torch.from_numpy(top_ids_np).to(self.device)
                res_z1 = bc_residual_adiabatic(
                    self.model, top_coords, top_ids,
                    torch.zeros(self.n_bc_side, device=self.device),
                    htc_t, tamb_t, tsv_t,
                    normal_dim=2, geom_scale=self.geom_scale,
                )
                face_residuals.append(res_z1)
            losses['bc_sides'] = bc_loss(torch.cat(face_residuals))

        return losses

    def _update_adaptive_weights(self, last_losses: Dict[str, float]) -> None:
        """Trigger NTK weight update using a single training scenario."""
        sc = self.train_data.scenarios[0]
        with torch.no_grad():
            pass  # just reuse last_losses for gradient norm estimation
        losses_tensor = {}
        htc_t = torch.tensor(sc.htc_norm, dtype=torch.float32, device=self.device)
        tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=self.device)
        tsv_t = torch.tensor(sc.tsv_frac, dtype=torch.float32, device=self.device)

        tim_k_t = torch.tensor(sc.tim_k_norm, dtype=torch.float32, device=self.device)
        T_pred = self.model(
            sc.coords, sc.layer_ids, sc.power, htc_t, tamb_t, tsv_t,
            region_ids=self._data_region_ids, tim_k_norm=tim_k_t,
        )
        losses_tensor['data'] = data_loss(T_pred, sc.temp)

        if 'pde' in last_losses and self.weights.pde > 0:
            n_ntk = min(len(self._col_coords), 5000)
            col_coords = self._col_coords[:n_ntk]
            col_ids = self._col_ids[:n_ntk]
            col_power = power_at_colloc_points(
                col_coords, col_ids,
                self.geometry, sc.power_blocks_wcm2 or {},
                sc.geom_extents,
                self.norm_stats.power_mean, self.norm_stats.power_std,
                self.device,
            )
            res = pde_residual(
                self.model, col_coords, col_ids,
                col_power,
                htc_t, tamb_t, tsv_t,
                self.layer_k, self.si_mask,
                self.norm_stats.T_min, self.norm_stats.T_max,
                self.geom_scale, self.norm_stats.power_std,
                col_k_lateral=self._col_k_lat[:n_ntk] if self._col_k_lat is not None else None,
                col_si_lateral=self._col_si_lat[:n_ntk] if self._col_si_lat is not None else None,
            )
            losses_tensor['pde'] = pde_loss(res)

        self.weights.update_ntk(self.model, losses_tensor)

    def _compute_lateral_k(
        self,
        col_coords: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (col_k_lateral, col_si_lateral) for 2p5d_stack geometries.

        For each point in a die-zone layer that falls outside all chiplet footprints,
        overrides k with underfill_k and marks the point as non-silicon so the
        temperature-dependent k(T) correction is not applied to underfill.

        Returns (None, None) for all other geometry types — zero overhead.
        """
        if (self.geometry.geometry_type != '2p5d_stack'
                or not self.geometry.die_footprints):
            return None, None

        L_x, L_y, _ = self.geom_scale
        coords_np = col_coords.detach().cpu().numpy()
        ids_np    = col_ids.cpu().numpy()
        x_phys    = coords_np[:, 0] * L_x
        y_phys    = coords_np[:, 1] * L_y

        # Start from per-layer defaults (same as the non-lateral path)
        k_np  = self.layer_k.cpu().numpy()[ids_np].copy()
        si_np = self.si_mask.cpu().numpy()[ids_np].copy()

        die_zone_indices = {
            i for i, layer in enumerate(self.geometry.layers)
            if any(dp.die_layer_name == layer.name
                   for dp in self.geometry.die_footprints)
        }

        for layer_i in die_zone_indices:
            in_layer = ids_np == layer_i
            if not in_layer.any():
                continue
            in_any = np.zeros(int(in_layer.sum()), dtype=bool)
            for dp in self.geometry.die_footprints:
                in_any |= (
                    (x_phys[in_layer] >= dp.x) &
                    (x_phys[in_layer] <  dp.x + dp.width) &
                    (y_phys[in_layer] >= dp.y) &
                    (y_phys[in_layer] <  dp.y + dp.height)
                )
            outside        = in_layer.copy()
            outside[in_layer] = ~in_any
            k_np[outside]  = self.geometry.underfill_k
            si_np[outside] = False

        return (
            torch.from_numpy(k_np).to(self.device),
            torch.from_numpy(si_np).to(self.device),
        )

    def _compute_region_ids(
        self,
        coords: torch.Tensor,   # (N, 3) normalised
    ) -> Optional[torch.Tensor]:
        """
        Return (N,) int tensor of chiplet region IDs for 2p5d_stack geometries.

        Region encoding: 0 = underfill/outside chiplets, 1 = first chiplet footprint,
        2 = second chiplet footprint (and so on for more chiplets).

        Returns None for non-2p5d geometries — zero overhead, model.forward ignores None.
        """
        if (self.geometry.geometry_type != '2p5d_stack'
                or not getattr(self.geometry, 'die_footprints', None)):
            return None

        L_x, L_y, _ = self.geom_scale
        coords_np = coords.detach().cpu().numpy()
        x_phys = coords_np[:, 0] * L_x
        y_phys = coords_np[:, 1] * L_y

        region_np = np.zeros(len(coords_np), dtype=np.int64)  # default: underfill (0)
        for region_id, dp in enumerate(self.geometry.die_footprints, start=1):
            in_fp = (
                (x_phys >= dp.x) & (x_phys < dp.x + dp.width) &
                (y_phys >= dp.y) & (y_phys < dp.y + dp.height)
            )
            region_np[in_fp] = region_id

        return torch.from_numpy(region_np).to(self.device)

    def _rar_update(self, epoch: int) -> None:
        """Append rar_add_n points to the persistent collocation set, chosen
        according to self.sampling_strategy:

          'rar'        — residual-proportional (original RAR-D; eps uniform floor)
          'hessian'    — curvature (Laplacian-trace) proportional, same eps floor
          'importance' — softmax-temperature resampling of the residual field
          'curriculum' — blends uniform -> residual-weighted over training
                         (blend factor grows from 0 at warmup to 1 at epochs=self.epochs)

        See src/pinn/sampling.py for the literature basis of each.
        """
        sc = self.train_data.scenarios[0]
        htc_t = torch.tensor(sc.htc_norm, dtype=torch.float32, device=self.device)
        tamb_t = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=self.device)
        tsv_t = torch.tensor(sc.tsv_frac, dtype=torch.float32, device=self.device)
        tim_k_t = torch.tensor(sc.tim_k_norm, dtype=torch.float32, device=self.device)

        cand_coords, cand_ids = sample_collocation_stratified(
            100_000, sc.geom_extents, self.geometry, self.device
        )
        cand_power = power_at_colloc_points(
            cand_coords, cand_ids,
            self.geometry, sc.power_blocks_wcm2 or {},
            sc.geom_extents,
            self.norm_stats.power_mean, self.norm_stats.power_std,
            self.device,
        )
        cand_k_lat, cand_si_lat = self._compute_lateral_k(cand_coords, cand_ids)
        cand_region_ids = self._compute_region_ids(cand_coords)

        # compute_adaptive_weights internally handles model.eval()/train() and
        # detaches its result — safe to call regardless of strategy.
        strategy_for_signal = 'hessian' if self.sampling_strategy == 'hessian' else 'rar'
        signal = compute_adaptive_weights(
            strategy_for_signal, self.model, cand_coords, cand_ids, cand_power,
            htc_t, tamb_t, tsv_t,
            self.layer_k, self.si_mask,
            self.norm_stats.T_min, self.norm_stats.T_max,
            self.geom_scale, self.norm_stats.power_std,
            col_k_lateral=cand_k_lat, col_si_lateral=cand_si_lat,
            region_ids=cand_region_ids, tim_k_norm=tim_k_t,
        )

        if self.sampling_strategy in ('rar', 'hessian'):
            eps = self.rar_epsilon
            weights = (1.0 - eps) * signal / (signal.sum() + 1e-12) + eps / len(signal)
            chosen = torch.multinomial(weights.float(), self.rar_add_n, replacement=False)
        elif self.sampling_strategy == 'importance':
            chosen = importance_resample(
                signal, self.rar_add_n, temperature=self.sampling_temperature,
                uniform_floor=self.rar_epsilon,
            )
        else:  # 'curriculum'
            blend = curriculum_blend_factor(epoch, self.epochs, self.curriculum_warmup_frac)
            chosen = curriculum_enhanced_resample(
                signal, self.rar_add_n, blend, temperature=self.sampling_temperature,
            )

        # .detach() defensively — cand_coords must never carry autograd history
        # into the persistent collocation buffer (see note in sampling.py's
        # compute_adaptive_weights about pde_residual's in-place requires_grad_).
        self._col_coords = torch.cat([self._col_coords, cand_coords[chosen].detach()])
        self._col_ids = torch.cat([self._col_ids, cand_ids[chosen]])
        if len(self._col_coords) > self.rar_max_col:
            self._col_coords = self._col_coords[-self.rar_max_col:]
            self._col_ids = self._col_ids[-self.rar_max_col:]

        # Refresh cached tensors after the collocation set grows
        self._col_k_lat, self._col_si_lat = self._compute_lateral_k(
            self._col_coords, self._col_ids
        )
        self._col_region_ids = self._compute_region_ids(self._col_coords)
        _log.info("[%s] epoch %d: +%d pts -> total col=%d",
                  self.sampling_strategy, epoch, self.rar_add_n, len(self._col_coords))

    def _validate(self) -> float:
        """Return mean absolute error in Kelvin on validation scenarios."""
        self.model.eval()
        all_errs = []
        T_range = self.norm_stats.T_max - self.norm_stats.T_min

        with torch.no_grad():
            for sc in self.val_data.scenarios:
                htc_t   = torch.tensor(sc.htc_norm,   dtype=torch.float32, device=self.device)
                tamb_t  = torch.tensor(sc.t_amb_norm, dtype=torch.float32, device=self.device)
                tsv_t   = torch.tensor(sc.tsv_frac,   dtype=torch.float32, device=self.device)
                tim_k_t = torch.tensor(sc.tim_k_norm, dtype=torch.float32, device=self.device)

                T_pred = self.model(
                    sc.coords, sc.layer_ids, sc.power, htc_t, tamb_t, tsv_t,
                    region_ids=self._data_region_ids, tim_k_norm=tim_k_t,
                )
                mae_norm = (T_pred - sc.temp).abs().mean().item()
                all_errs.append(mae_norm * T_range)

        return float(np.mean(all_errs))

    def _log_epoch(
        self, epoch: int, losses: Dict[str, float], val_mae: float, elapsed: float
    ) -> None:
        msg = (
            f"Epoch {epoch:5d}/{self.epochs} | "
            f"loss={losses.get('data', 0):.4e} "
            f"pde={losses.get('pde', 0):.4e} "
            f"bc={losses.get('bc_top', 0):.4e} | "
            f"val_MAE={val_mae:.3f} K | "
            f"col={len(self._col_coords)} | "
            f"{elapsed:.0f}s"
        )
        _log.info(msg)

        self.history['epoch'].append(epoch)
        self.history['train_total'].append(sum(losses.values()))
        self.history['train_data'].append(losses.get('data', 0))
        self.history['train_pde'].append(losses.get('pde', 0))
        self.history['train_bc'].append(losses.get('bc_top', 0))
        self.history['val_mae_K'].append(val_mae)

        if self._writer:
            for k, v in losses.items():
                self._writer.add_scalar(f'train/{k}', v, epoch)
            self._writer.add_scalar('val/mae_K', val_mae, epoch)
            self._writer.add_scalar('weights/pde', self.weights.pde, epoch)
            self._writer.add_scalar('weights/bc_top', self.weights.bc_top, epoch)

    def _save_checkpoint(self, epoch: int, val_mae: float, best: bool) -> None:
        fname = f"{self.geometry.name}_best.pt" if best else f"{self.geometry.name}_final.pt"
        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_mae_K': val_mae,
            'geometry_name': self.geometry.name,
            'norm_stats': {
                'T_min': self.norm_stats.T_min,
                'T_max': self.norm_stats.T_max,
                'power_mean': self.norm_stats.power_mean,
                'power_std': self.norm_stats.power_std,
                'geom_extents': self.norm_stats.geom_extents,
                'htc_min': self.norm_stats.htc_min,
                'htc_max': self.norm_stats.htc_max,
                't_amb_min': self.norm_stats.t_amb_min,
                't_amb_max': self.norm_stats.t_amb_max,
            },
            'history': self.history,
        }
        torch.save(ckpt, self.output_dir / fname)
