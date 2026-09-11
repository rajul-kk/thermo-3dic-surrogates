"""Loader for IC-ThermBench S2-S5, reproducing their split exactly."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

log = logging.getLogger(__name__)

TRAIN_RATIO = 0.8      # data_provider/data_factory.py::TRAIN_RATIO
VAL_FRACTION = 0.9     # data_provider/data_loader.py::split_train_val_test step 3

SCOPES = {
    'level2': ['chiplet_power', 'grid_x', 'grid_y'],
    'level3': ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k'],
    'level4': ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k',
               'ambient_K', 'h_w_m2k', 'r_convec_k_per_w'],
    'level5': ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k',
               'ambient_K', 'h_w_m2k', 'r_convec_k_per_w'],
}

# Channels that are constant within a sample and therefore usable as scalars.
SCALAR_CHANNELS = {'ambient_K', 'h_w_m2k', 'r_convec_k_per_w'}


@dataclass
class Split:
    """One scope's data. X is (B, X, Y, Z, P); Y is (B, X, Y, Z), kelvin."""
    scope: str
    channels: List[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def summary(self) -> str:
        return (f"{self.scope}: train={len(self.x_train)} val={len(self.x_val)} "
                f"test={len(self.x_test)} P={len(self.channels)} "
                f"grid={self.y_train.shape[1:]}")


def load_mat_pair(folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and transpose one scope's .mat pair. Mirrors their load_mat_pair()."""
    with h5py.File(folder / 'input.mat', 'r') as f:
        x = f['data'][()]
    with h5py.File(folder / 'output.mat', 'r') as f:
        y = f['data'][()]

    if x.ndim != 5:
        raise ValueError(f'input should be 5-D [B,P,Z,Y,X], got {x.shape}')
    if y.ndim != 4:
        raise ValueError(f'output should be 4-D [B,Z,Y,X], got {y.shape}')
    if x.shape[0] != y.shape[0]:
        raise ValueError(f'batch mismatch: {x.shape[0]} vs {y.shape[0]}')
    if tuple(x.shape[2:]) != tuple(y.shape[1:]):
        raise ValueError(f'spatial mismatch: {x.shape[2:]} vs {y.shape[1:]}')

    x = np.transpose(x, (0, 4, 3, 2, 1))   # (B,P,Z,Y,X) -> (B,X,Y,Z,P)
    y = np.transpose(y, (0, 3, 2, 1))      # (B,Z,Y,X)   -> (B,X,Y,Z)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


def split_indices(total: int, train_ratio: float = TRAIN_RATIO) -> Dict[str, slice]:
    """Their index split, as slices. No shuffling, by design."""
    trainval_total = int(total * train_ratio)
    trainval_total = max(2, min(trainval_total, total - 1))
    n_train = int(trainval_total * VAL_FRACTION)
    n_train = max(1, min(n_train, trainval_total - 1))
    return {
        'train': slice(0, n_train),
        'val':   slice(n_train, trainval_total),
        'test':  slice(trainval_total, total),
    }


def load_scope(data_root: Path, scope: str, split_data: bool = True) -> Split:
    """Load one scope."""
    if scope not in SCOPES:
        raise ValueError(f'unknown scope {scope!r}; expected one of {sorted(SCOPES)}')
    folder = data_root / f'{scope}_steady'
    x, y = load_mat_pair(folder)

    if not split_data:
        empty_x = x[:0]
        empty_y = y[:0]
        return Split(scope, SCOPES[scope], empty_x, empty_y, empty_x, empty_y, x, y)

    idx = split_indices(len(x))
    return Split(
        scope, SCOPES[scope],
        x[idx['train']], y[idx['train']],
        x[idx['val']],   y[idx['val']],
        x[idx['test']],  y[idx['test']],
    )


def scalar_channel_indices(channels: List[str]) -> Dict[str, int]:
    """Map of scalar-channel name -> C-axis index, for the channels that are scalars."""
    return {name: i for i, name in enumerate(channels) if name in SCALAR_CHANNELS}


def spatial_channel_indices(channels: List[str]) -> Dict[str, int]:
    """Map of spatial-channel name -> C-axis index."""
    return {name: i for i, name in enumerate(channels) if name not in SCALAR_CHANNELS}


def verify_against_upstream(data_root: Path, scope: str, upstream_repo: Path) -> bool:
    """Check our arrays match theirs exactly, using their code as the reference."""
    import sys
    sys.path.insert(0, str(upstream_repo))
    from data_provider.data_loader import load_mat_pair as their_load  # noqa: E402
    from data_provider.data_loader import split_train_val_test as their_split  # noqa: E402

    their_x, their_y, _, _ = their_load(str(data_root / f'{scope}_steady'))
    tx, ty, vx, vy, sx, sy = their_split(
        their_x, their_y, num_trajectories=-1, train_ratio=TRAIN_RATIO)

    ours = load_scope(data_root, scope)
    pairs = [
        ('train x', ours.x_train, tx), ('train y', ours.y_train, ty),
        ('val x',   ours.x_val,   vx), ('val y',   ours.y_val,   vy),
        ('test x',  ours.x_test,  sx), ('test y',  ours.y_test,  sy),
    ]
    for name, mine, theirs in pairs:
        theirs_np = theirs.detach().cpu().numpy()
        assert mine.shape == theirs_np.shape, (
            f'{scope} {name}: shape {mine.shape} vs upstream {theirs_np.shape}')
        assert np.array_equal(mine, theirs_np), (
            f'{scope} {name}: values differ from upstream '
            f'(max abs diff {np.abs(mine - theirs_np).max():.6g})')
        log.info('%s %s: exact match, shape %s', scope, name, mine.shape)
    return True
