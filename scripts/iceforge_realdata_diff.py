"""Run `iceforge diff` on REAL archived layout-randomised scenarios (g4-g6), pre-snap vs snapped offsets.
Stack built with the same ICESimulator path as scripts/resolve_from_metadata.py. Nothing under data/ is modified.
Usage: python scripts/iceforge_realdata_diff.py [--n 8] [--geoms 4 5 6] [--r 4] [--only g4:3:orig]
"""
import argparse
import ast
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools' / 'iceforge'))
os.environ['ICEFORGE_GUARD'] = '0'          # the guard would refuse the unsnapped stacks, which is the point here
from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import place_chiplets, snap_offsets
from src.simulators.ice_simulator import ICESimulator
from iceforge.check import check_stack
from iceforge.model import parse_stk
from iceforge.diff import run_diff, strip_private

SUFFIX = ''
ICE_EXE = 'wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator'
ARCH = ROOT / 'data' / '_archive_pre_snap_20260930'
CORR = ROOT / 'data'
OUT = ROOT / 'results' / 'iceforge_realdata'
# work dir must have no spaces (WSL path handling)
WORK = Path(os.environ.get('ICEFORGE_WORK', r'C:\Users\rajul\AppData\Local\Temp\claude\d--Work-3D-ICE-Thermal-modelling-Thermo\7f6b7d72-6072-4d1c-b3a6-41b1ce88b1c3\scratchpad\realdata'))


def offsets(m):
    offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
            for k in m if k.startswith('placement_dx_')}
    return {('' if k == 'blocks' else k): v for k, v in offs.items()}


def pick(g, n):
    files = sorted((ARCH / f'3d-ice-layout-geometry{g}').rglob('*.npz'))
    idx = np.unique(np.linspace(0, len(files) - 1, n).round().astype(int))
    return [files[i] for i in idx]


def corrected_path(f):
    return CORR / f.parent.parent.name / f.parent.name / f.name


def build_and_diff(f, arm, r, nz=None):
    d = np.load(f, allow_pickle=True)
    m = d['metadata'].item()
    offs = offsets(m)
    base = get_geometry_by_name(m['geometry'])
    info = {}
    if arm == 'snapped':
        snapped = snap_offsets(base, offs, margin_um=0.0)
        if snapped is None:
            raise ValueError('no grid-aligned placement')
        offs = snapped
        co = offsets(np.load(corrected_path(f), allow_pickle=True)['metadata'].item())
        info['snapped_matches_corrected_metadata'] = all(
            abs(snapped[k][0] - co[k][0]) < 1e-6 and abs(snapped[k][1] - co[k][1]) < 1e-6 for k in snapped)
    geom = place_chiplets(base, offs)
    blocks = {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks}
    scen = {'power_blocks': blocks, 'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
            'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
    wd = WORK / f'{f.stem}_{arm}'
    shutil.rmtree(wd, ignore_errors=True)
    cfg, res = wd / 'c', wd / 'o'
    cfg.mkdir(parents=True)
    res.mkdir()
    sim = ICESimulator(cfg, res, ICE_EXE)
    sim.generate_config_files(geom, scen)
    stk = str(cfg / 'stack.stk')
    info['check_error_codes'] = sorted({f_.code for f_ in check_stack(parse_stk(stk)) if f_.severity == 'error'})
    t0 = time.time()
    rep = run_diff(stk, r=r, nz=nz, backend='wsl', exe=ICE_EXE, outdir=str(wd / 'ice'), edge_source='own',
                   log=lambda *a: None)
    sec = time.time() - t0
    out = strip_private(rep, max_cells=60)
    layers = rep['layers']
    t_amb = float(m['t_ambient_celsius']) + 273.15
    nfl = sum(c['n_flagged'] for c in layers)
    ned = sum(c['n_flagged_die_edge'] for c in layers)
    nadj = sum(c['n_flagged_edge_adjacent'] for c in layers)
    def agg(src, key):
        return sum(c['by_edge_source'][src][key] for c in layers)
    by = {src: dict(n_flagged=agg(src, 'n_flagged'), n_die_edge=agg(src, 'n_die_edge'),
                    n_edge_adjacent=agg(src, 'n_edge_adjacent'), n_near_edge=agg(src, 'n_near_edge'),
                    n_interior=agg(src, 'n_interior')) for src in ('own', 'union')}
    for src in ('own', 'union'):
        h = {}
        for c in layers:
            for k, v in c['by_edge_source'][src]['distance_hist'].items():
                h[k] = h.get(k, 0) + v
        by[src]['distance_hist'] = h
    np.savez_compressed(OUT / 'cases' / f'{f.stem}_{arm}{SUFFIX}_fields.npz',
                        **{f"{c['instance']}_{k}": c['_fields'][k] for c in layers for k in ('dT', 'flag')})
    row = dict(file=f.stem, geometry=m['geometry'], arm=arm, verdict=rep['verdict'], r=r, nz=nz,
               verdict_own_preregistered=rep['verdict_by_edge_source']['own'],
               verdict_union_posthoc=rep['verdict_by_edge_source']['union'], by_edge_source=by,
               ice_peak_K=max(c['max_rise_ice'] for c in layers) + t_amb,
               ice_peak_rise_K=max(c['max_rise_ice'] for c in layers),
               ref_peak_rise_K=max(c['max_rise_ref'] for c in layers),
               max_abs_dT=max(c['max_abs_dT'] for c in layers),
               n_layers=len(layers), n_layers_disagree=sum(c['verdict'] != 'AGREE' for c in layers),
               n_flagged=nfl, n_flagged_die_edge=ned, n_flagged_edge_adjacent=nadj,
               frac_edge_or_adjacent=(ned + nadj) / nfl if nfl else None,
               frac_die_edge_only=ned / nfl if nfl else None,
               seconds=round(sec, 1), **info)
    out['summary_row'] = row
    (OUT / 'cases' / f'{f.stem}_{arm}{SUFFIX}.json').write_text(json.dumps(out, indent=1, default=float))
    shutil.rmtree(wd, ignore_errors=True)
    return row


def sanity(f, arm, row):
    ref = f if arm == 'orig' else corrected_path(f)
    peak = float(np.load(ref, allow_pickle=True)['temp'].max())
    return dict(npz_peak_K=peak, peak_mismatch_K=row['ice_peak_K'] - peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--geoms', type=int, nargs='+', default=[4, 5, 6])
    ap.add_argument('--r', type=int, default=4)
    ap.add_argument('--nz', type=int, default=None, help='reference sub-layers per layer (default: <=25 um each)')
    ap.add_argument('--only', default=None, help='gN:index:arm, quick test')
    ap.add_argument('--suffix', default='', help='case file suffix (spot checks)')
    a = ap.parse_args()
    global SUFFIX
    SUFFIX = a.suffix
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases').mkdir(exist_ok=True)
    rows = []
    for g in a.geoms:
        for i, f in enumerate(pick(g, a.n)):
            for arm in ('orig', 'snapped'):
                if a.only and a.only != f'g{g}:{i}:{arm}':
                    continue
                cj = OUT / 'cases' / f'{f.stem}_{arm}{SUFFIX}.json'
                cached = json.loads(cj.read_text())['summary_row'] if cj.exists() and not a.only else None
                if cached and 'by_edge_source' in cached and cached.get('r') == a.r and cached.get('nz') == a.nz:
                    row = cached
                else:     # stale (pre-2026-10-06 classification) or missing: recompute and overwrite
                    row = build_and_diff(f, arm, a.r, a.nz)
                row.update(sanity(f, arm, row))
                row['case_index'] = i
                fe = row['frac_edge_or_adjacent']
                print(f"g{g} {i} {f.stem} {arm}: {row['verdict']} ice {row['ice_peak_rise_K']:.3f} ref {row['ref_peak_rise_K']:.3f} "
                      f"maxdT {row['max_abs_dT']:.3f} flagged {row['n_flagged']} (edge+adj {fe}) "
                      f"peakmismatch {row['peak_mismatch_K']:+.3f} K  {row['seconds']}s", flush=True)
                rows.append(row)
    if a.only:
        return
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=1, default=float))
    L = ['# iceforge diff on real archived g4-g6 data (r=%d, nz=%s)' % (a.r, a.nz), '',
         "Two classifications of the same solves. `own` = pre-registered rule (die-edge mask from each layer's own floorplan/layout edges).",
         "`union` = POST-HOC rule (in-plane union of all layers' edges), added after the own rule labelled the artefact's vertical spill-over as interior.", '',
         '## Verdict counts', '', '| geometry | arm | n | own AGREE | own die-edge | own diffuse | union AGREE (post-hoc) | union die-edge (post-hoc) | union diffuse (post-hoc) |',
         '|---|---|---|---|---|---|---|---|---|']
    for g in sorted({r['geometry'] for r in rows}):
        for arm in ('orig', 'snapped'):
            rr = [r for r in rows if r['geometry'] == g and r['arm'] == arm]
            cnt = lambda key, v: sum(r[key] == 'DISAGREE (%s)' % v if v != 'AGREE' else r[key] == 'AGREE' for r in rr)
            L.append(f"| {g} | {arm} | {len(rr)} | " + ' | '.join(
                str(cnt(k, v)) for k in ('verdict_own_preregistered', 'verdict_union_posthoc') for v in ('AGREE', 'die-edge', 'diffuse')) + ' |')
    L += ['', '## Pooled distance of unexplained flagged cells to the nearest union edge (cells, chessboard), orig arm', '']
    for arm in ('orig', 'snapped'):
        h = {}
        for r in rows:
            if r['arm'] == arm:
                for k, v in r['by_edge_source']['union']['distance_hist'].items():
                    h[k] = h.get(k, 0) + v
        L.append(f"- {arm}: {dict(sorted(h.items(), key=lambda kv: (kv[0] in ('>=8', 'no-edge'), kv[0].zfill(3))))}")
    npass = sum(abs(r['peak_mismatch_K']) <= 0.05 for r in rows)
    L += ['', f'## Sanity: 3D-ICE peak vs archived/corrected npz peak (|diff| <= 0.05 K): {npass}/{len(rows)} pass', '',
          '## Per case', '',
          '| geom | scenario | arm | verdict own | verdict union (post-hoc) | 3D-ICE peak rise K | ref peak rise K | max abs dT K | flagged (own) | edge+adj (own) | edge+adj (union) | near-edge (union) | interior (union) | peak vs npz K |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        o, u = r['by_edge_source']['own'], r['by_edge_source']['union']
        L.append(f"| {r['geometry']} | {r['file'].split('_', 1)[-1]} | {r['arm']} | {r['verdict_own_preregistered']} | {r['verdict_union_posthoc']} | "
                 f"{r['ice_peak_rise_K']:.3f} | {r['ref_peak_rise_K']:.3f} | {r['max_abs_dT']:.3f} | {o['n_flagged']} | "
                 f"{o['n_die_edge'] + o['n_edge_adjacent']} | {u['n_die_edge'] + u['n_edge_adjacent']} | {u['n_near_edge']} | {u['n_interior']} | {r['peak_mismatch_K']:+.3f} |")
    (OUT / 'summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main()
