"""Run the maximum-principle lint (L005) on PUBLIC thermal-surrogate datasets stored as MATLAB v7.3 / HDF5 tensors.

Supported layouts (both use HDF5 key 'data'; channel 0 of the input is the power field):
  thermfm   input (N, P, L, H, W)  output (N, L, H, W)   Therm-FM steady release, channel 0 = volumetric power density
  ictherm   input (N, P, Z, Y, X)  output (N, Z, Y, X)   IC-ThermBench S2-S5, channel 0 = chiplet power; channel 3
                                                          (S3-S5) = local thermal conductivity
Both are the same tensor layout; the names only select the default report label.

Limits (read before interpreting):
  * No geometry or floorplan is released, so L006 (off-grid footprint) cannot be run. For ictherm we report a
    proxy: powered cells whose conductivity equals the dataset minimum (a heated gap-material cell).
  * L005 is only informative where some cells are unpowered. If every cell has power > 0 the check is vacuous;
    the report states the unpowered fraction so that case is visible.
  * The check is applied to the whole released volume (all layers together, the hottest node anywhere against the
    hottest powered node anywhere). Layers not released (spreader, sink) cannot be checked.

Usage: python scripts/lint_external.py NAME=DIR[@fmt] [...] [--max-samples N] [--json results/lint_external.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.validation.lint import MAX_PRINCIPLE_TOL_K, check_arrays, check_max_principle  # noqa: E402


def read_pair(folder, fmt='thermfm'):
    """Open input.mat / output.mat lazily. Returns (h5 file in, h5 file out, input dset, output dset)."""
    import h5py
    folder = Path(folder)
    fi, fo = h5py.File(folder / 'input.mat', 'r'), h5py.File(folder / 'output.mat', 'r')
    di, do = fi['data'], fo['data']
    if di.ndim != 5 or do.ndim != 4 or di.shape[0] != do.shape[0] or di.shape[2:] != do.shape[1:]:
        raise ValueError(f'unexpected shapes input {di.shape}, output {do.shape}')
    return fi, fo, di, do


def sample_excess(power, temp):
    """Max-principle excess in K: hottest node minus hottest powered node, if the hottest node is unpowered (else 0).
    Also returns the L005 finding list from the project linter, so both agree."""
    powered = power > 0
    if not powered.any():
        return 0.0, []
    findings = check_max_principle(temp.ravel(), power.ravel())
    i = np.unravel_index(int(temp.argmax()), temp.shape)
    excess = float(temp[i] - temp[powered].max()) if not powered[i] else 0.0
    return max(excess, 0.0), findings


def hot_distance(power, temp):
    """Distance in cells (Euclidean, in-plane, same layer) from the hottest node to the nearest powered node of that
    layer. Small values (1-2) mean the hot unpowered cell sits at the edge of a powered block."""
    from scipy.ndimage import distance_transform_edt
    l, y, x = np.unravel_index(int(temp.argmax()), temp.shape)
    if not (power[l] > 0).any():
        return float('nan')
    return float(distance_transform_edt(power[l] <= 0)[y, x])


def lint_pair(folder, fmt='thermfm', max_samples=None, batch=200, transpose_power=False):
    fi, fo, di, do = read_pair(folder, fmt)
    n = di.shape[0] if not max_samples else min(max_samples, di.shape[0])
    has_k = fmt == 'ictherm' and di.shape[1] >= 4
    k_min = float(np.min(di[:min(n, 500), 3])) if has_k else None
    viol, excesses, unpowered_frac, bad_arrays, gap_heated = [], [], [], 0, []
    dists, hot_k_at_min = [], []
    for s in range(0, n, batch):
        P = np.asarray(di[s:s + batch, 0], float)
        T = np.asarray(do[s:s + batch], float)
        K = np.asarray(di[s:s + batch, 3], float) if has_k else None
        if transpose_power:                                     # swap the in-plane axes of the power (and k) fields
            P = P.swapaxes(-1, -2)
            K = K.swapaxes(-1, -2) if has_k else None
        for j in range(len(P)):
            p, t = P[j], T[j]
            coords = np.zeros((p.size, 3))                      # arrays-only rule: coordinates are not needed
            if any(f.code == 'L001' for f in check_arrays(coords, t.ravel(), p.ravel())):
                bad_arrays += 1
                continue
            unpowered_frac.append(float((p <= 0).mean()))
            ex, f = sample_excess(p, t)
            if f:
                viol.append(s + j)
                excesses.append(ex)
                dists.append(hot_distance(p, t))
                if has_k:
                    hot_k_at_min.append(bool(K[j][np.unravel_index(int(t.argmax()), t.shape)] <= k_min * (1 + 1e-6)))
            if has_k:
                gap = (K[j] <= k_min * (1 + 1e-6)) & (p > 0)   # powered cells holding the dataset's lowest conductivity
                gap_heated.append(float(gap.mean()))
    fi.close(), fo.close()
    e = np.array(excesses)
    out = {'folder': str(folder), 'format': fmt, 'shape_input': list(di.shape), 'power_transposed': transpose_power, 'samples_checked': n - bad_arrays,
           'samples_bad_arrays_L001': bad_arrays, 'samples_with_L005': len(viol),
           'frac_with_L005': len(viol) / max(n - bad_arrays, 1),
           'excess_K': ({'min': float(e.min()), 'median': float(np.median(e)), 'max': float(e.max())} if len(e) else None),
           'hot_node_distance_to_powered_cell': ({'median': float(np.nanmedian(dists)), 'max': float(np.nanmax(dists)),
                                                  'frac_within_2_cells': float(np.mean(np.array(dists) <= 2))}
                                                 if dists else None),
           'first_violating_indices': viol[:10],
           'mean_unpowered_cell_fraction': float(np.mean(unpowered_frac)) if unpowered_frac else None,
           'samples_with_no_unpowered_cell': int(sum(u == 0 for u in unpowered_frac)),
           'tolerance_K': MAX_PRINCIPLE_TOL_K}
    if has_k:
        out['k_min'] = k_min
        out['violations_hot_node_at_k_min_fraction'] = float(np.mean(hot_k_at_min)) if hot_k_at_min else None
        out['mean_powered_cells_at_k_min_fraction'] = float(np.mean(gap_heated)) if gap_heated else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('datasets', nargs='+', help='NAME=DIR or NAME=DIR@fmt (fmt thermfm|ictherm, default thermfm)')
    ap.add_argument('--max-samples', type=int)
    ap.add_argument('--transpose-power', action='store_true', help='swap in-plane axes of the power/k fields first')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    res = {}
    for spec in a.datasets:
        name, _, rest = spec.partition('=')
        folder, _, fmt = rest.partition('@')
        fmt = fmt or 'thermfm'
        t0 = time.time()
        r = lint_pair(folder, fmt, a.max_samples, transpose_power=a.transpose_power)
        r['seconds'] = round(time.time() - t0, 1)
        res[name] = r
        print(f"{name}: {r['samples_checked']} samples, {r['samples_with_L005']} with L005, excess {r['excess_K']}, "
              f"unpowered cells {r['mean_unpowered_cell_fraction']:.3f}, hot-node distance {r['hot_node_distance_to_powered_cell']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
