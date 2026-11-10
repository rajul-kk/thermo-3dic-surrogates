"""Nonlinear regime A, verification item (iii) of docs/protocol_nonlinear_A.md section 3: `iceforge diff --with-hotspot` on
the final linear solve of 10 scenarios (3D-ICE vs iceforge's independent reference solver vs HotSpot).

Selection, fixed before any of these runs: the first 2 converged scenarios by name in each of the five splits
(train, iid, ood_gain, ood_law, near_critical). Each final linear solve is rebuilt exactly as scripts/gen_nonlinear_A.py
ran it (same geometry, HTC, ambient and delivered block powers at the fixed point), so 3D-ICE's fresh peak must also
match the stored field.

Reference resolution, fixed before the 10 runs: --r 1 (reference at 100 x 100 and 2r = 200 x 200 cells laterally) and
--nz 8 sub-layers per layer. iceforge's defaults (r = 4, <= 25 um per sub-layer, i.e. 200 sub-layers in the 5 mm heat
sink) ran out of memory on this machine in a one-scenario trial. diff reports the reference's own r -> 2r change, so the
coarser reference's discretisation error is stated next to each comparison.

HotSpot: the SuperLU build (~/HotSpot_slu/hotspot in WSL), run through ~/hs_capped/hotspot (the same binary under a 6 GB address-space
cap), on a 64 x 64 grid (--hotspot-grid 64). iceforge's default HotSpot, ~/HotSpot/hotspot, is built without SuperLU; its self-test times
out on these stiff stacks and the HotSpot leg was missing (first attempt, 2026-11-10). At iceforge's default grid (128 x 128) the SuperLU
factorisation passed 7 GB and exhausted the machine (second attempt); at 64 x 64 it takes ~1 min and 3 GB. 3D-ICE and the reference keep
the resolution above. Deviation recorded in docs/protocol_nonlinear_A.md.

usage: python scripts/check_nlA_iii.py [--only NAME ...] [--resume]   ->  results/nonlinear_A/check_iii.json (+ .log)
  --resume keeps the scenarios already in check_iii.json and runs the rest (the file is rewritten after every scenario)
"""
import argparse, json, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
import gen_nonlinear_A as G  # noqa: E402  (same geometry, base scenario and simulator set-up)

OUT = REPO / 'results' / 'nonlinear_A'
SPLITS = ['train', 'iid', 'ood_gain', 'ood_law', 'near_critical']
ICE = '/home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator'
HOTSPOT = 'wsl /home/rajul/hs_capped/hotspot'


def selection(plan):
    have = {p.stem for p in G.DATA.glob('*.npz')}
    out = []
    for s in SPLITS:
        names = sorted(it['name'] for it in plan if it['split'] == s and it['name'] in have)
        out += [it for n in names[:2] for it in plan if it['name'] == n]
    return out


def write_stack(it, tmp):
    sim = G.sim_in(tmp)
    sc = G.base_sc(it['htc'], it['t_ambient'], it['delivered'])
    sim.generate_config_files(G.GEOM, sc)
    stk = sorted((tmp / 'cfg').rglob('*.stk'))
    if len(stk) != 1:
        raise RuntimeError(f'expected one .stk, found {stk}')
    return stk[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    plan = json.loads((OUT / 'plan.json').read_text())
    sel = selection(plan)
    if a.only:
        sel = [it for it in sel if it['name'] in a.only]
    out = OUT / ('check_iii.json' if not a.only else 'check_iii_trial.json')
    rows = json.loads(out.read_text()) if a.resume and out.exists() else []
    done = {r['name'] for r in rows}
    for it in sel:
        if it['name'] in done:
            continue
        tmp = Path(tempfile.mkdtemp(prefix='nlA_iii_'))
        try:
            stk = write_stack(it, tmp)
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, '-m', 'iceforge', 'diff', str(stk), '--with-hotspot', '--json',
                                '--exe', ICE, '--backend', 'wsl', '--hotspot-exe', HOTSPOT, '--hotspot-grid', '64', '-o', str(tmp / 'ice'), '--r', '1', '--nz', '8'],
                               cwd=REPO / 'tools' / 'iceforge', capture_output=True, text=True)
            secs = time.perf_counter() - t0
            try:
                rep = json.loads(r.stdout[r.stdout.index('{'):])
            except ValueError:
                raise RuntimeError(f'iceforge diff gave no JSON (exit {r.returncode}): {r.stdout[-600:]} {r.stderr[-600:]}')
            z = np.load(G.DATA / f"{it['name']}.npz", allow_pickle=True)
            row = dict(name=it['name'], split=it['split'], htc=it['htc'], exit=r.returncode, seconds=round(secs, 1),
                       stored_peak_K=float(z['temp'].max()), report=rep)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'report'}), rep.get('verdict'),
                  'hotspot', (rep.get('hotspot') or {}).get('status'), flush=True)
            out.write_text(json.dumps(rows, indent=1))  # after every scenario, so a killed run keeps what it finished
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    out.write_text(json.dumps(rows, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
