"""M3b sample generation (docs/protocol_mesh_convergence.md section 10). Modes: g5 | gen <worker> <n_workers>."""
import copy, json, math, sys, time, importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).parent
mode = sys.argv[1] if len(sys.argv) > 1 else 'g5'
args = sys.argv[2:]
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location('ms', HERE / 'mesh_sweeps.py'); ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
R, parse_stk, OUT = ms.R, ms.parse_stk, ms.OUT
from iceforge.model import Rect
SDIR = OUT / 'm3b' / 'samples'; SDIR.mkdir(parents=True, exist_ok=True)

model = R.build_model(parse_stk(str(ms.BASE / 'stack.stk')))
isrc = ms.src_layer(model)
dens_b = [d for rc, d in model.layers[isrc].power_rects if abs(rc.x - 1000) < 1e-6 and abs(rc.y - 1000) < 1e-6][0]
nzf = ms.nz_fine(12.5)
FAM = {  # (id, N range, s range, m range)
    'F0': (0, (1, 2), (25, 200), (3, 30)), 'T0': (1, (1, 2), (25, 200), (3, 30)), 'T1': (2, (1, 1), (250, 400), (3, 30)),
    'T2': (3, (4, 6), (25, 200), (3, 30)), 'T3': (4, (1, 2), (25, 200), (30, 60))}
JOBS = [('F0', i) for i in range(280)] + [(f, i) for f in ('T0', 'T1', 'T2', 'T3') for i in range(40)]

def draw(fam, idx):
    fid, (n0, n1), (s0, s1), (m0, m1) = FAM[fam]
    rng = np.random.default_rng((20261201, fid, idx))
    spots = []
    for _ in range(int(rng.integers(n0, n1 + 1))):
        s = float(math.exp(rng.uniform(math.log(s0), math.log(s1)))); m = float(math.exp(rng.uniform(math.log(m0), math.log(m1))))
        spots.append((float(rng.uniform(2000.0, 3000.0 - s)), float(rng.uniform(2000.0, 3000.0 - s)), s, m))
    return spots

def with_spots(spots):
    m2 = copy.copy(model); layers = list(model.layers); L = copy.copy(layers[isrc])
    L.power_rects = list(L.power_rects) + [(Rect(x, y, s, s), (m - 1.0) * dens_b) for x, y, s, m in spots]
    layers[isrc] = L; m2.layers = layers
    return m2

def fields(m2, spots, n_win=2):
    t0 = time.perf_counter(); coarse = R.solve_window(m2, ms.WHOLE(m2), 1, nz=nzf); t_coarse = time.perf_counter() - t0
    return coarse, t_coarse

def central(F, x_start):                       # F covers [x_start, ...] at 12.5 um; return the central [2000, 3000]^2 block
    i0 = int(round((2000.0 - x_start) / 12.5)); return F[i0:i0 + 80, i0:i0 + 80]

def make(fam, idx):
    path = SDIR / f'{fam}_{idx:03d}.npz'
    if path.exists(): return
    spots = draw(fam, idx); m2 = with_spots(spots)
    coarse, t_coarse = fields(m2, spots)
    w2 = R.solve_window(m2, (1500.0, 3500.0, 1500.0, 3500.0), 8, nz=nzf, coarse=coarse, bc='dirichlet')
    truth = central(w2.layer_field(isrc), 1500.0)
    t0 = time.perf_counter(); w1 = R.solve_window(m2, (2000.0, 3000.0, 2000.0, 3000.0), 8, nz=nzf, coarse=coarse, bc='dirichlet')
    classical = w1.layer_field(isrc); t_cl = time.perf_counter() - t0
    Fc = coarse.layer_field(isrc)[19:31, 19:31]
    g = (np.arange(96) + 0.5) / 8.0 - 0.5
    cu = map_coordinates(Fc, np.meshgrid(g, g, indexing='ij'), order=1, mode='nearest')
    xe = 1900.0 + 12.5 * np.arange(97)
    dmap = dens_b * R.rect_overlap_area(xe, xe, Rect(1000.0, 1000.0, 3000.0, 3000.0)) / (12.5 ** 2)
    for x, y, s, m in spots:
        dmap = dmap + (m - 1.0) * dens_b * R.rect_overlap_area(xe, xe, Rect(x, y, s, s)) / (12.5 ** 2)
    np.savez(path, x_coarse_up=cu.astype(np.float64), x_pow=(dmap / dens_b).astype(np.float32), y_truth=truth, y_classical=classical,
             coarse_up_center=cu[8:88, 8:88], t_classical=t_cl, t_coarse=t_coarse, spots=json.dumps(spots), fam=fam, idx=idx)

if mode == 'g5':
    rows = []
    for idx in range(5):
        spots = draw('F0', idx); m2 = with_spots(spots); coarse, _ = fields(m2, spots)
        a = central(R.solve_window(m2, (1500.0, 3500.0, 1500.0, 3500.0), 8, nz=nzf, coarse=coarse, bc='dirichlet').layer_field(isrc), 1500.0)
        b = central(R.solve_window(m2, (1000.0, 4000.0, 1000.0, 4000.0), 8, nz=nzf, coarse=coarse, bc='dirichlet').layer_field(isrc), 1000.0)
        rows.append(dict(idx=idx, n_spots=len(spots), max_abs_2mm_vs_3mm_K=float(np.abs(a - b).max()), peak_2mm=float(a.max()), peak_3mm=float(b.max())))
        print('G5', rows[-1], flush=True)
    ok = max(r['max_abs_2mm_vs_3mm_K'] for r in rows) <= 0.005
    (OUT / 'm3b' / 'g5.json').write_text(json.dumps(dict(rows=rows, pass_=ok), indent=1)); print('G5 PASS' if ok else 'G5 FAIL', flush=True)
else:
    w, nw = int(args[0]), int(args[1])
    mine = [j for k, j in enumerate(JOBS) if k % nw == w]
    t0 = time.perf_counter()
    for n, (fam, idx) in enumerate(mine):
        try: make(fam, idx)
        except Exception as e: print('ERR', fam, idx, repr(e)[:150], flush=True)
        if n % 10 == 0: print(f'worker {w}: {n + 1}/{len(mine)} elapsed {time.perf_counter() - t0:.0f}s', flush=True)
    print('GEN DONE', w, flush=True)
