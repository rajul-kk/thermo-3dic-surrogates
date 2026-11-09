"""Did ThermoNO learn the 3D-ICE die-edge artefact? (report §9.32; A1 and A2)

ThermoNO is trained on the PRE-fix layout data (die edges mid-cell; spurious peaks in unpowered edge cells), with the
same seed, folds and defaults as §9.26, then
  A1  scored on its own held-out PRE-fix scenarios for artefact reproduction: is its hottest node an unpowered cell,
      is it the same node as the (spurious) true peak, and how far above the hottest powered node does it sit?
      Controls: the training-free backbone (true physics) and the pre-fix 3D-ICE truth itself.
  A2  applied unchanged to the same held-out scenarios re-solved on the grid-aligned placement (POST-fix inputs and
      truth): what training on the artefact costs, against the backbone and against ThermoNO trained on post-fix data
      (`results/thermono_snap/`, same seed and folds).
Usage: python scripts/artefact_learning.py geometry5 [--old-root data/_archive_pre_snap_20260930] [--threads 4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.hotspot_eval import hotspot_metrics, kfold_indices          # noqa: E402
from scripts.layout_cv import per_scenario_stats                         # noqa: E402
from scripts.thermono_train import build, fit_fold, predict, sample_points  # noqa: E402


def node_power(data_root, g):
    """Per-node power density of every scenario, in build()'s (sorted-file) order."""
    files = sorted(Path(data_root, f'3d-ice-layout-{g}').rglob(f'{g}_*.npz'))
    return [f.stem for f in files], [np.load(f, allow_pickle=True)['power'].astype(np.float64) for f in files]


def artefact(p, P, true_argmax=None):
    i = int(p.argmax())
    out = {'argmax_unpowered': bool(P[i] <= 0), 'excess_K': float(p.max() - p[P > 0].max())}
    if true_argmax is not None:
        out['same_node_as_truth'] = bool(i == true_argmax)
    return out


def summarise(rows, keys):
    s = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        s[k] = float(np.mean(v)) if k not in ('hotspot_loc_err_um',) else float(np.median(v))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geometry')
    ap.add_argument('--old-root', default='data/_archive_pre_snap_20260930')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--folds', nargs='*', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=300, help='smoke tests only; §9.26 uses 300')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    # §9.26 defaults, exactly as scripts/thermono_train.py
    args = argparse.Namespace(epochs=a.epochs, ch=16, blocks=3, modes=16, batch=4, lr=3e-3, w_top=4.0, patience=8, seed=0,
                              geo_arch='mlp', geo_attn=False, local_k=1, multiscale=0, device='cpu')
    g = a.geometry
    D_old, D_new = build(g, a.old_root), build(g, 'data')
    names_old, P_old = node_power(a.old_root, g)
    names_new, P_new = node_power('data', g)
    assert names_old == names_new, 'pre/post-fix scenario lists differ'
    n = D_old['q'].shape[0]
    # resumable: rows are saved after every fold, and finished folds are skipped on restart
    part = Path(f'results/artefact_learning/{g}_seed0{a.tag}.partial.json')
    part.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(part.read_text()) if part.exists() and a.epochs == 300 else []
    done = {r['fold'] for r in rows}
    for fi, te in enumerate(kfold_indices(n, 5, args.seed)):
        if fi not in a.folds or fi in done:
            continue
        t = time.time()
        tr = np.setdiff1d(np.arange(n), te)
        model, qs, ts, eps = fit_fold(D_old, tr, args)
        f_old = predict(D_old, model, te, qs, ts, args)          # A1: its own held-out pre-fix scenarios
        f_new = predict(D_new, model, te, qs, ts, args)          # A2: same scenarios, post-fix inputs
        for j, i in enumerate(te):
            y_old, y_new = D_old['temp'][i], D_new['temp'][i]
            r = {'scenario': names_old[i], 'fold': fi,
                 'truth_old': artefact(y_old, P_old[i]), 'truth_new': artefact(y_new, P_new[i])}
            ta = int(y_old.argmax())
            for m in ('backbone', 'thermono'):
                p_old = sample_points(D_old, f_old[m][j], i)
                p_new = sample_points(D_new, f_new[m][j], i)
                r[f'{m}_old'] = {**artefact(p_old, P_old[i], ta), **per_scenario_stats(y_old, p_old),
                                 **hotspot_metrics(p_old, y_old, D_old['coords'][i])}
                r[f'{m}_on_new'] = {**artefact(p_new, P_new[i]), **per_scenario_stats(y_new, p_new),
                                    **hotspot_metrics(p_new, y_new, D_new['coords'][i])}
            rows.append(r)
        part.write_text(json.dumps(rows, default=float))
        print(f'fold {fi}: {eps} epochs, {time.time() - t:.0f}s', flush=True)

    keys = ['argmax_unpowered', 'excess_K', 'same_node_as_truth', 'r2', 'det_mae', 'hotspot_loc_err_um',
            'top1pct_recall', 'abs_peak_temp_err_K']
    summary = {'n': len(rows)}
    for col in ('truth_old', 'truth_new', 'backbone_old', 'thermono_old', 'backbone_on_new', 'thermono_on_new'):
        summary[col] = summarise([r[col] for r in rows], keys)
    snap = Path(f'results/thermono_snap/thermono_{g}_seed0.json')
    if snap.exists():
        summary['thermono_trained_new'] = json.loads(snap.read_text())['summary']['thermono']
    out = Path(f'results/artefact_learning/{g}_seed0{a.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
