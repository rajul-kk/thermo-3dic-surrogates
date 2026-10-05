"""Projection test of the 3D-ICE die-edge artefact (statistic: src/validation/artefact_projection.py).

For architecture A, geometry g, fold f and seed s: train A on the OLD labels (O_s) and on the NEW labels (N_s) with the
same inputs, hyper-parameters and seed; predict the held-out scenarios with every model and score L = mean O - mean N (and
the single-seed O_s - N_s) against the true artefact D = T_old - T_new, with same-label seed-pair nulls.
FNO / CNO-FNO training = scripts/fno_fair_cv.py protocol (as in the earlier fa_core.py), ThermoNO = scripts/thermono_train.py
defaults (section 9.26).  Folds = kfold_indices(n, 5, 0) over the name-sorted scenarios.
Roots hold `3d-ice-layout-<geometry>/` sub-directories.  Resumable: one partial JSON per (arch, geometry), updated per fold.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.hotspot_eval import kfold_indices                                  # noqa: E402
from src.validation import artefact_projection as ap                            # noqa: E402

# minutes per single training on a T4 (FNO/CNO: results/fno_artefact timings; ThermoNO: a guess, no log has timings)
PRIOR_MIN = {('fno', 'geometry4'): 3.5, ('fno', 'geometry5'): 2.9, ('fno', 'geometry6'): 4.4,
             ('cno-fno-attn', 'geometry4'): 8.0, ('cno-fno-attn', 'geometry5'): 6.7, ('cno-fno-attn', 'geometry6'): 11.1,
             ('thermono', 'geometry4'): 2.0, ('thermono', 'geometry5'): 2.0, ('thermono', 'geometry6'): 2.0}
SEEDS = {'fno': (0, 1, 2), 'thermono': (0, 1, 2), 'cno-fno-attn': (0, 1)}
FNO_HP = dict(ch=32, blocks=4, modes=[16, 16, 8], epochs=300, min_epochs=200, patience=40, batch=2, lr=1e-3)
THERMO_HP = dict(epochs=300, ch=16, blocks=3, modes=16, batch=4, lr=3e-3, w_top=4.0, patience=8, geo_arch='mlp',
                 geo_attn=False, local_k=1, multiscale=0)


# ---------------------------------------------------------------- FNO family (verbatim from the earlier fa_core.py)
def _fno_imports():
    global get_geometry_by_name, points_to_grid, FNODataset, build_cno_fno, build_fno, compute_norm_stats
    from src.core.geometry_builders import get_geometry_by_name
    from src.core.mesh import points_to_grid
    from src.fno.data_loader import FNODataset
    from src.fno.model import build_cno_fno, build_fno
    from src.pinn.data_loader import compute_norm_stats


def fno_build(variant, grid, a, dev):
    if variant == 'fno':
        return build_fno(grid, modes=tuple(a['modes']), hidden_ch=a['ch'], n_blocks=a['blocks'], device=dev)
    return build_cno_fno(grid, ch=a['ch'], n_fno_blocks=a['blocks'], n_cno_layers=2, use_attention=True, n_heads=4, device=dev)


def fno_stack(ds, files, norm, dev):
    it = ds.items
    t_rng = norm.T_max - norm.T_min
    amb = np.array([float(np.load(f, allow_pickle=True)['metadata'].item().get('t_ambient_kelvin',
                    float(np.load(f, allow_pickle=True)['metadata'].item()['t_ambient_celsius']) + 273.15))
                    for f in files], dtype=np.float32)
    T = torch.stack([x['T_norm'] for x in it]) * t_rng + norm.T_min
    rise = T - torch.as_tensor(amb)[:, None, None, None]
    X = [torch.stack([x[k] for x in it]).to(dev) for k in ('Q_norm', 'layer_id_norm', 'htc_norm', 't_amb_norm', 'tsv_frac')]
    return X, rise.to(dev), torch.as_tensor(amb, device=dev)


def fno_loss(out, tgt):
    d = lambda z: z - z.mean(dim=(1, 2, 3), keepdim=True)
    return ((out - tgt) ** 2).mean() + ((d(out) - d(tgt)) ** 2).mean()


def fno_train(variant, grid, geom, fit_f, val_f, a, dev, seed):
    norm = compute_norm_stats(fit_f, {geom: get_geometry_by_name(geom)})
    Xf, Rf, _ = fno_stack(FNODataset(fit_f, norm, grid), fit_f, norm, dev)
    Xv, Rv, _ = fno_stack(FNODataset(val_f, norm, grid), val_f, norm, dev)
    scale = float(Rf.std())
    torch.manual_seed(seed)
    model = fno_build(variant, grid, a, dev)
    steps = a['epochs'] * int(np.ceil(len(fit_f) / a['batch']))
    opt = torch.optim.AdamW(model.parameters(), lr=a['lr'], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a['lr'], total_steps=steps)
    best, best_state, stale = np.inf, None, 0
    fwd = lambda X, idx: model(*[x[idx] for x in X])
    for ep in range(a['epochs']):
        model.train()
        for b in torch.randperm(len(fit_f), device=dev).split(a['batch']):
            opt.zero_grad()
            fno_loss(fwd(Xf, b), Rf[b] / scale).backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([fno_loss(fwd(Xv, torch.tensor([i], device=dev)), Rv[i:i + 1] / scale).item()
                               for i in range(len(val_f))]))
        if v < best - 1e-7:
            best, best_state, stale = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= a['patience'] and ep + 1 >= a['min_epochs']:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, norm, scale, ep + 1


def fno_predict(model, norm, scale, files, grid, dev):
    """Flat grid-order prediction and truth (K) per file, plus the grid-order node coordinates."""
    X, R, A = fno_stack(FNODataset(files, norm, grid), files, norm, dev)
    out = []
    with torch.no_grad():
        for j, f in enumerate(files):
            o = model(*[x[torch.tensor([j], device=dev)] for x in X])[0] * scale
            pred = (o + A[j]).cpu().numpy().ravel().astype(np.float64)
            true = (R[j] + A[j]).cpu().numpy().ravel().astype(np.float64)
            c = np.load(f, allow_pickle=True)['coords'].astype(np.float64)
            cg = np.stack(points_to_grid(c, c[:, 0], c[:, 1], c[:, 2]), -1).reshape(-1, 3)
            out.append((pred, true, cg))
    return out


def list_files(root, g):
    return sorted(Path(root, f'3d-ice-layout-{g}').rglob(f'{g}_*.npz'), key=lambda p: p.name)


def fno_unit(variant, g, te, ctx, seeds, a, dev, max_train):
    """One fold: (pred_old {seed: [arr per test scenario]}, pred_new, D list, epochs)."""
    fo, fn, grid = ctx['fo'], ctx['fn'], ctx['grid']
    tr = np.setdiff1d(np.arange(len(fo)), te)
    if max_train:
        tr = tr[:max_train]
    rng = np.random.default_rng(0 + ctx['fold'] + 1)          # fno_fair_cv numbers folds from 1 for its val split
    val_idx = set(rng.choice(len(tr), size=max(3, len(tr) // 9), replace=False).tolist())
    sel = lambda files, want: [files[i] for k, i in enumerate(tr) if (k in val_idx) == want]
    P, eps, truth, coords = {'O': {}, 'N': {}}, {}, {}, {}
    for arm, files in (('O', fo), ('N', fn)):
        te_f = [files[i] for i in te]
        for s in seeds:
            model, norm, scale, e = fno_train(variant, grid, g, sel(files, False), sel(files, True), a, dev, s)
            pr = fno_predict(model, norm, scale, te_f, grid, dev)
            P[arm][s] = [p[0] for p in pr]
            truth[arm], coords[arm] = [p[1] for p in pr], pr[0][2]
            eps[f'{arm}{s}'] = e
            del model
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
    assert np.allclose(coords['O'], coords['N']), 'old/new node coordinates differ'
    return P['O'], P['N'], [yo - yn for yo, yn in zip(truth['O'], truth['N'])], eps


# ---------------------------------------------------------------- ThermoNO
def thermo_unit(g, te, ctx, seeds, hp, dev):
    from scripts.thermono_train import fit_fold, predict, sample_points
    Do, Dn = ctx['Do'], ctx['Dn']
    tr = np.setdiff1d(np.arange(Do['q'].shape[0]), te)
    P, eps = {'O': {}, 'N': {}}, {}
    for arm, D in (('O', Do), ('N', Dn)):
        for s in seeds:
            args = argparse.Namespace(**hp, seed=s, device=str(dev))
            model, qs, ts, e = fit_fold(D, tr, args)
            f = predict(D, model, te, qs, ts, args)['thermono']
            P[arm][s] = [sample_points(D, f[j], i) for j, i in enumerate(te)]
            eps[f'{arm}{s}'] = e
            del model
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
    return P['O'], P['N'], [Do['temp'][i] - Dn['temp'][i] for i in te], eps


def make_ctx(arch, g, old_root, new_root):
    fo, fn = list_files(old_root, g), list_files(new_root, g)
    assert [p.name for p in fo] == [p.name for p in fn], 'old/new scenario lists differ'
    ctx = {'names': [p.stem for p in fo], 'n': len(fo)}
    if arch == 'thermono':
        from scripts.thermono_train import build
        Do, Dn = build(g, old_root), build(g, new_root)
        assert np.array_equal(Do['coords'], Dn['coords']), 'old/new node coordinates differ'
        # ThermoNO's inputs are FV-grid fields built from the placement offsets in the metadata, and those (unlike the node
        # `power` arrays) DO change under the snap.  Keep the inputs identical for both arms (the corrected-data inputs) and
        # swap only the labels: the OLD-label arm sees Dn's inputs with Do's 3D-ICE temperature labels.
        Dmix = {**Dn, 'theta': Do['theta'], 'mask': Do['mask'], 'temp': Do['temp']}
        ctx.update(Do=Dmix, Dn=Dn, input_diff_unsnapped={k: float(np.abs(Do[k] - Dn[k]).max()) for k in ('q', 'bb', 'kl', 'kv', 'htc', 'tamb')},
                   input_diff={k: float(np.abs(Dmix[k] - Dn[k]).max()) for k in ('q', 'bb', 'kl', 'kv', 'htc', 'tamb')})
        return ctx
    _fno_imports()
    c0 = np.load(fo[0])['coords']
    grid = tuple(len(np.unique(c0[:, i])) for i in (1, 0, 2))
    norm = compute_norm_stats([fo[0]], {g: get_geometry_by_name(g)})
    io, in_ = FNODataset([fo[0]], norm, grid).items[0], FNODataset([fn[0]], norm, grid).items[0]
    diff = {k: float((io[k] - in_[k]).abs().max()) for k in ('Q_norm', 'layer_id_norm', 'htc_norm', 't_amb_norm', 'tsv_frac')}
    ctx.update(fo=fo, fn=fn, grid=grid, input_diff=diff)
    return ctx


def run_all(old_root, new_root, out_dir, plan, folds, dev, meta=None, budget_h=11.4, epochs=None, max_train=None,
            seeds_override=None, t_start=None):
    t_start = t_start or time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fhp, thp = dict(FNO_HP), dict(THERMO_HP)
    if epochs:
        fhp.update(epochs=epochs, min_epochs=min(fhp['min_epochs'], epochs))
        thp['epochs'] = max(epochs, 5)          # fit_fold validates every 5th epoch
    print('PLAN: arch, geometry, seeds, estimated hours on a T4 (2 arms x seeds trainings per fold)', flush=True)
    total = 0
    for arch, g in plan:
        sd = seeds_override or SEEDS[arch]
        est = 2 * len(sd) * PRIOR_MIN[(arch, g)] * len(folds) / 60
        total += est
        print(f'  {arch:<13} {g}  seeds {list(sd)}  {est:.1f} h for {len(folds)} folds', flush=True)
    print(f'  total {total:.1f} h (budget {budget_h} h; folds that would overrun it are skipped)', flush=True)
    for arch, g in plan:
        sd = tuple(seeds_override or SEEDS[arch])
        part = out_dir / f'{arch}_{g}.partial.json'
        state = json.loads(part.read_text()) if part.exists() else {}
        todo = [fi for fi in range(5) if fi in folds and str(fi) not in state]
        if not todo:
            continue
        ctx = make_ctx(arch, g, old_root, new_root)
        print(f'{arch} {g}: {ctx["n"]} scenarios, model-input max|old-new| {ctx["input_diff"]}'
              + (f' (ThermoNO inputs of the raw old/new files differ: {ctx["input_diff_unsnapped"]})' if 'input_diff_unsnapped' in ctx else ''), flush=True)
        if True:
            state['input_diff'] = ctx['input_diff']
            state['input_diff_unsnapped'] = ctx.get('input_diff_unsnapped')
        folds_idx = kfold_indices(ctx['n'], 5, 0)
        for fi in todo:
            te = folds_idx[fi]
            if (time.time() - t_start) + 2 * len(sd) * PRIOR_MIN[(arch, g)] * 60 > budget_h * 3600:
                print(f'SKIP {arch} {g} fold {fi}: would exceed the {budget_h} h budget', flush=True)
                continue
            ctx['fold'] = fi
            t = time.time()
            if arch == 'thermono':
                Po, Pn, D, eps = thermo_unit(g, te, ctx, sd, thp, dev)
            else:
                Po, Pn, D, eps = fno_unit(arch, g, te, ctx, sd, fhp, dev, max_train)
            names = [ctx['names'][i] for i in te]
            recs = {}
            for j, nm in enumerate(names):
                recs[nm] = ap.scenario_record({s: Po[s][j] for s in sd}, {s: Pn[s][j] for s in sd}, D[j])
                recs[nm]['D_absmax_K'] = float(np.abs(D[j]).max())
            state[str(fi)] = {'records': recs, 'epochs': eps, 'seconds': time.time() - t}
            part.write_text(json.dumps(state, default=float))
            if not (out_dir / f'{arch}_{g}_sample.npz').exists():          # 2 scenarios, float16, for sanity plots
                h = lambda x: np.asarray(x, np.float16)
                smp = {}
                for j in range(min(2, len(names))):
                    mo, mn = np.mean([Po[s][j] for s in sd], 0), np.mean([Pn[s][j] for s in sd], 0)
                    smp.update({f'{names[j]}__D': h(D[j]), f'{names[j]}__L': h(mo - mn),
                                f'{names[j]}__L_seed{sd[0]}': h(Po[sd[0]][j] - Pn[sd[0]][j]),
                                f'{names[j]}__null_nn': h(Pn[sd[0]][j] - Pn[sd[-1]][j])})
                np.savez_compressed(out_dir / f'{arch}_{g}_sample.npz', **smp)
            b = {k: round(float(np.nanmean([r['mean'][k] for r in recs.values()])), 3) for k in ap.FIELDS}
            print(f'{arch} {g} fold {fi}: {(time.time() - t) / 60:.1f} min, epochs {eps}, fold-mean {b} '
                  f'| elapsed {(time.time() - t_start) / 3600:.2f} h', flush=True)
            write_summary(out_dir, meta, plan)
        ctx.clear()
    return write_summary(out_dir, meta, plan)


def write_summary(out_dir, meta, plan):
    out_dir = Path(out_dir)
    summ = {'meta': meta or {}, 'decision_rule': ap.RULE, 'support_K': ap.SUPPORT_K, 'results': {}}
    for arch, g in plan:
        part = out_dir / f'{arch}_{g}.partial.json'
        if not part.exists():
            continue
        state = json.loads(part.read_text())
        done = sorted(int(k) for k in state if k.isdigit())
        recs = {}
        for k in done:
            recs.update(state[str(k)]['records'])
        names = sorted(recs)
        r = [recs[nm] for nm in names]
        summ['results'][f'{arch}|{g}'] = {
            'folds_done': done, 'n_scenarios': len(r), 'input_diff': state.get('input_diff'), 'input_diff_unsnapped': state.get('input_diff_unsnapped'),
            'seconds_per_fold': {str(k): state[str(k)]['seconds'] for k in done},
            'epochs': {str(k): state[str(k)]['epochs'] for k in done},
            'analysis': {f: ap.analyse(r, f) for f in ap.FIELDS},
            'per_scenario': {nm: {k: ({f: recs[nm][k][f] for f in ap.FIELDS} if k == 'mean' else
                                      {f: [x[f] for x in recs[nm][k]] for f in ap.FIELDS})
                                  for k in ('mean', 'single', 'null_nn', 'null_oo')} for nm in names}}
    (out_dir / 'projection_summary.json').write_text(json.dumps(summ, default=float))
    return summ


def default_plan(geoms=('geometry5', 'geometry6'), extras=True):
    plan = [('fno', g) for g in geoms] + [('thermono', g) for g in geoms]
    if extras:
        plan += [('cno-fno-attn', g) for g in geoms] + [('fno', 'geometry4'), ('thermono', 'geometry4')]
    return plan
