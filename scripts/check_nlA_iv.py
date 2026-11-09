"""Nonlinear regime A, verification item (iv) of docs/protocol_nonlinear_A.md section 3: energy balance < 1e-4 on every
converged file, with src/validation/energy.py (heat leaving the heat-sink face, from the exported bottom-cell
temperatures through half a cell and the HTC, against the power field 3D-ICE was given).

usage: python scripts/check_nlA_iv.py  ->  results/nonlinear_A/energy_balance.json
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.validation.energy import energy_balance  # noqa: E402

DATA = REPO / 'data' / '3d-ice-leakage-A' / 'geometry1'
LIMIT = 1e-4

rows = []
for f in sorted(DATA.glob('*.npz')):
    m = np.load(f, allow_pickle=True)['metadata'][0]
    if not bool(m.get('picard_converged', True)):
        continue
    r = energy_balance(f, include_feedback=True)
    rows.append(dict(file=f.name, split=m['split'], htc=float(m['htc']), imbalance=r['ratio'] - 1.0,
                     meta_over_field=r['meta_over_field'] - 1.0, P_W=r['P_field_W']))
imb = np.array([r['imbalance'] for r in rows])
summary = dict(files=len(rows), limit=LIMIT, n_pass=int((np.abs(imb) < LIMIT).sum()),
               median_abs=float(np.median(np.abs(imb))), p99_abs=float(np.percentile(np.abs(imb), 99)),
               max_abs=float(np.abs(imb).max()), mean_signed=float(imb.mean()),
               by_htc={str(h): dict(n=int(s.sum()), n_pass=int((np.abs(imb[s]) < LIMIT).sum()),
                                    mean_signed=float(imb[s].mean()), sd=float(imb[s].std()),
                                    max_abs=float(np.abs(imb[s]).max()))
                       for h in sorted({r['htc'] for r in rows}) for s in [np.array([r['htc'] == h for r in rows])]},
               max_abs_power_meta_vs_field=float(max(abs(r['meta_over_field']) for r in rows)))
out = REPO / 'results' / 'nonlinear_A' / 'energy_balance.json'
out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1))
print(json.dumps(summary, indent=1))
