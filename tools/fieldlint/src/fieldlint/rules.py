"""Rules F001-F009.

Every rule is a function `rule(ds: Dataset, opt: Options) -> RuleResult` and carries a docstring that states (a) why the
check is physically justified and (b) when it does NOT apply. Rules never raise on bad data: they report. A rule that
cannot run (missing field, wrong problem class) returns status 'skipped' with the reason.

Scope: steady diffusion / heat-type problems  -div(k grad u) = s  with u held at (or relaxed towards) a boundary level.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SEVERITIES = ('error', 'warning', 'info')


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    samples: List[str] = field(default_factory=list)       # ids of (some of) the affected samples


@dataclass
class RuleResult:
    code: str
    name: str
    status: str = 'ok'                                      # ok | flagged | skipped
    findings: List[Finding] = field(default_factory=list)
    summary: str = ''                                       # one line, always present
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Options:
    max_samples: Optional[int] = 400        # evenly spaced subsample used by per-sample rules (None = all)
    # F002
    source_rel_threshold: float = 1e-6      # source cell counts as "powered" if source > this * max(source)
    max_principle_rel_tol: float = 1e-3     # excess tolerance, fraction of the field's range
    max_principle_abs_tol: float = 0.0
    neighbourhood: int = 1                  # in-plane dilation radius (cells) of the source mask
    neighbourhood_z: int = 0                # dilation radius through the thickness (3D fields)
    # F003
    smooth_sigma: float = 2.0               # Gaussian sigma in cells
    orientation_margin: float = 0.10        # alternative must beat the stored correlation by this much
    # F008 / F009
    op_margin: float = 0.05                 # an alternative input orientation must beat the stored R^2 by this much
    op_r2_applicable: float = 0.30          # below this best-orientation median R^2 the operator model does not describe the field
    op_shifts: int = 3                      # also try circular shifts of the inputs by +-1..op_shifts cells along each axis (0 = off)
    op_min_samples: int = 3
    op_face_mean: str = 'auto'              # face conductivity: 'auto' (fitted per dataset), 'harmonic', 'arithmetic' or 'geometric'
    op_face_samples: int = 32               # samples used to choose the face mean when 'auto'
    min_inplane_cells: int = 8              # scope gate: skip rules F002/F003/F008/F009 when an in-plane axis is shorter
    pair_samples: int = 32                  # F009: number of samples scored pairwise (M x M matrix)
    pair_tie: float = 0.10                  # F009: the diagonal counts as best when within this R^2 of the row maximum
    pair_r2_applicable: float = 0.30        # F009: skip when the median row-maximum R^2 is below this
    pair_warn: float = 0.25                 # F009: fraction of rows with a better partner that gives a warning
    pair_error: float = 0.50                # ... and an error
    # F004
    dup_rtol: float = 1e-5                  # near-duplicate: equal after quantising to dup_rtol * range
    # F006
    linear_max_samples: int = 1500
    linear_max_features: int = 1024
    linear_seed: int = 0
    # F007
    energy_tol: float = 1e-2
    # F005
    max_temperature_k: float = 1500.0


# ── helpers ──────────────────────────────────────────────────────────────────────
def _short(ids, n=5):
    return list(ids[:n])


def gaussian_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing of the last two axes (edge-padded). numpy only."""
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    out = np.asarray(a, float)
    for axis in (-2, -1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode='edge')
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for j, wj in enumerate(w):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(j, j + n)
            acc += wj * p[tuple(sl)]
        out = acc
    return out


def dilate(mask: np.ndarray, r_inplane: int = 1, r_z: int = 0) -> np.ndarray:
    """Chebyshev dilation of a boolean mask: r_inplane cells along the last two axes, r_z along axis 0 of a 3D mask."""
    out = mask.copy()
    for axis, r in ((-2, r_inplane), (-1, r_inplane)) + (((0, r_z),) if mask.ndim == 3 else ()):
        if r <= 0:
            continue
        acc = out.copy()
        n = out.shape[axis]
        for s in range(1, r + 1):
            if s >= n:
                break
            a = [slice(None)] * out.ndim
            b = [slice(None)] * out.ndim
            a[axis], b[axis] = slice(s, None), slice(None, n - s)
            acc[tuple(a)] |= out[tuple(b)]
            acc[tuple(b)] |= out[tuple(a)]
        out = acc
    return out


def _variants(a: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """The orientations tested by F003, applied to the last two axes. 'transpose' is None for non-square grids."""
    sq = a.shape[-1] == a.shape[-2]
    return {'stored': a,
            'transpose': np.swapaxes(a, -1, -2) if sq else None,
            'flip_x': a[..., ::-1],
            'flip_y': a[..., ::-1, :],
            'flip_xy': a[..., ::-1, ::-1]}


def _pooled_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation after removing the per-layer mean (a, b are ([z,] y, x))."""
    ax = tuple(range(a.ndim - 2, a.ndim))
    a = a - a.mean(axis=ax, keepdims=True)
    b = b - b.mean(axis=ax, keepdims=True)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


def scope_skip(ds, opt: Options) -> Optional[str]:
    """Reason why the steady 2D diffusion rules do not apply to this dataset, or None.

    Out of scope: datasets flagged time_dependent (trajectory snapshots, not steady solutions) and fields that are effectively
    1D (an in-plane axis shorter than `min_inplane_cells`, e.g. a 1 x N line or a 3 x N strip)."""
    if getattr(ds, 'time_dependent', False):
        return 'out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution'
    shp = None
    if ds.field_shapes:
        shp = (ds.field_shapes() or {}).get('u')
    if shp is None:
        for _i, s in ds.samples(1):
            shp = s.u.shape
    if shp is not None and len(shp) >= 2 and min(shp[-2:]) < opt.min_inplane_cells:
        return (f'out of scope: the field is effectively 1D (in-plane shape {tuple(shp[-2:])}, an axis is shorter than '
                f'{opt.min_inplane_cells} cells); the steady 2D diffusion premises do not hold')
    return None


# ── F001 ─────────────────────────────────────────────────────────────────────────
def f001_integrity(ds, opt: Options) -> RuleResult:
    """F001 finite values and consistent shapes.

    Why: every other rule indexes the fields cell by cell, so u, source, k and the masks must share one grid, and a NaN/Inf
    silently poisons losses and normalisation statistics. Shapes must also agree across samples for batching.
    Does NOT apply: datasets that legitimately mix grid sizes (variable-resolution benchmarks) - the cross-sample shape
    message is then expected and can be ignored (it is a warning, not an error).
    """
    res = RuleResult('F001', 'integrity')
    fs = ds.field_shapes() if ds.field_shapes else {}
    ushape = fs.get('u')
    for nm, shp in fs.items():
        if nm != 'u' and ushape is not None and tuple(shp) != tuple(ushape):
            res.findings.append(Finding('F001', 'error', f"field '{nm}' has per-sample shape {tuple(shp)} but u has "
                                        f"{tuple(ushape)}; the fields do not share a grid", {'shapes': {k: list(v) for k, v in fs.items()}}))
    if ushape is not None and len(ushape) not in (2, 3):
        res.findings.append(Finding('F001', 'error', f'u must be 2D or 3D per sample, got shape {tuple(ushape)}'))
    shapes, nonfinite, neg_mask = {}, {}, 0
    n = 0
    for i in ds.indices(opt.max_samples):
        s = ds.get(i)
        if s is None:
            continue
        n += 1
        shapes.setdefault(tuple(s.u.shape), []).append(s.id or str(i))
        for nm in ('u', 'source', 'k'):
            a = getattr(s, nm)
            if a is None:
                continue
            if a.shape != s.u.shape:
                res.findings.append(Finding('F001', 'error', f"sample {s.id}: '{nm}' shape {a.shape} != u shape {s.u.shape}", {}, [s.id]))
            bad = int((~np.isfinite(a)).sum()) if a.dtype.kind == 'f' else 0
            if bad:
                nonfinite.setdefault(nm, []).append(s.id or str(i))
        if s.spacing is not None and len(s.spacing) != s.u.ndim:
            res.findings.append(Finding('F001', 'error', f'sample {s.id}: spacing has {len(s.spacing)} entries for a {s.u.ndim}D field', {}, [s.id]))
    if ds.errors:
        ids = [str(i) for i in sorted(ds.errors)]
        first = ds.errors[sorted(ds.errors)[0]]
        res.findings.append(Finding('F001', 'error', f'{len(ids)} sample(s) could not be loaded; first: sample {ids[0]}: {first}',
                                    {'count': len(ids)}, _short(ids)))
    for nm, ids in nonfinite.items():
        res.findings.append(Finding('F001', 'error', f"{len(ids)} of {n} checked sample(s) hold non-finite values in '{nm}'",
                                    {'count': len(ids), 'checked': n}, _short(ids)))
    if len(shapes) > 1:
        res.findings.append(Finding('F001', 'warning', f'samples have {len(shapes)} different u shapes: '
                                    f'{ {str(k): len(v) for k, v in shapes.items()} }', {}, _short(next(iter(shapes.values())))))
    res.status = 'flagged' if res.findings else 'ok'
    res.summary = f'{n} samples checked; ' + ('no problems' if not res.findings else f'{len(res.findings)} problem(s)')
    res.metrics = {'checked': n, 'load_failures': len(ds.errors)}
    return res


# ── F002 ─────────────────────────────────────────────────────────────────────────
def f002_max_principle(ds, opt: Options) -> RuleResult:
    """F002 maximum principle.

    Why: for steady diffusion  -div(k grad u) = s  with s >= 0 everywhere (no interior sinks), u cannot have a strict
    interior maximum away from the sources: the largest value is attained where s > 0 or on the Dirichlet boundary. A field
    whose hottest cell is far from every source therefore means u, source and k disagree (transposed or shifted inputs,
    layer-blind power field, a label from another sample). The allowed zone is the source mask dilated by
    `neighbourhood` cells in-plane (`neighbourhood_z` through the thickness) to absorb discretisation, plus any
    dirichlet_mask. The excess is (max u) - (max u over the allowed zone), tested against a tolerance relative to the
    field's range. Reports the fraction of violating samples.

    Does NOT apply: time-dependent datasets and effectively 1D fields (skipped as out of scope, see scope_skip); problems with sinks (s < 0, e.g. Peltier cooling or radiation modelled as a source) - samples with
    negative source are skipped; time-dependent or advection-dominated fields; fields where every cell is powered (the
    check is vacuous and reported as such); datasets whose source is a coarse proxy of the true heating (e.g. a
    block-averaged power map on a finer grid) - raise `neighbourhood`.
    """
    res = RuleResult('F002', 'max-principle')
    why = scope_skip(ds, opt)
    if why:
        res.status, res.summary = 'skipped', why
        return res
    n = viol = vac = sinks = 0
    excess, ids = [], []
    for i, s in ds.samples(opt.max_samples):
        if s.source is None:
            res.status, res.summary = 'skipped', 'no source field'
            return res
        u, src = np.asarray(s.u, float), np.asarray(s.source, float)
        if not (np.isfinite(u).all() and np.isfinite(src).all()):
            continue
        smax = src.max()
        thr = opt.source_rel_threshold * smax
        if smax <= 0:
            vac += 1
            continue
        if src.min() < -thr:
            sinks += 1
            continue
        on = src > thr
        if on.mean() > 1 - 1e-3:
            vac += 1
            continue
        allowed = dilate(on, opt.neighbourhood, opt.neighbourhood_z)
        if s.dirichlet_mask is not None:
            allowed = allowed | np.asarray(s.dirichlet_mask, bool)
        rng = float(u.max() - u.min())
        tol = max(opt.max_principle_abs_tol, opt.max_principle_rel_tol * rng)
        if rng <= 0 or (s.ambient is not None and u.max() <= s.ambient + tol):
            vac += 1                                         # nothing is heated above the boundary level
            continue
        n += 1
        ex = float(u.max() - u[allowed].max())
        if ex > tol:
            viol += 1
            excess.append(ex)
            ids.append(s.id or str(i))
    res.metrics = {'checked': n, 'violating': viol, 'fraction': viol / n if n else None,
                   'vacuous_or_no_heating': vac, 'skipped_negative_source': sinks}
    if excess:
        res.metrics['excess'] = {'median': float(np.median(excess)), 'max': float(np.max(excess))}
        res.metrics['first_violating'] = _short(ids, 10)
    if n == 0:
        res.status = 'skipped'
        res.summary = f'nothing to check ({vac} vacuous / no-heating, {sinks} with sinks)'
        return res
    frac = viol / n
    res.summary = f'{viol}/{n} samples ({100 * frac:.1f}%) have their maximum away from every source'
    if viol:
        res.status = 'flagged'
        sev = 'error' if frac >= 0.2 else 'warning'
        res.findings.append(Finding('F002', sev, f'{viol} of {n} samples ({100 * frac:.1f}%) have the hottest cell more than '
                            f'{opt.neighbourhood} cell(s) from any source and {np.median(excess):.3g} (median) above the hottest '
                            f'allowed cell: violates the maximum principle; u, source and k probably disagree', res.metrics, _short(ids)))
    return res


# ── F003 ─────────────────────────────────────────────────────────────────────────
def f003_orientation(ds, opt: Options) -> RuleResult:
    """F003 orientation / alignment of source versus u.

    Why: temperature is positively correlated with a spatially smoothed version of the heat source (heat diffuses out of
    the source). If the source is stored transposed or flipped relative to u (a row-major vs column-major export, a
    swapped x/y convention) the correlation is much lower as stored than for the correct orientation. For each sample the
    Pearson correlation (per-layer means removed) between Gaussian-smoothed source and u is computed as stored, transposed,
    and flipped along x, y and both. A sample votes for an alternative only when it beats the stored orientation on two
    measures: the plain smoothed correlation by `orientation_margin`, and the band-passed correlation (both fields minus a
    4x wider Gaussian, which removes large flat regions and keeps the hot spots) by half that margin. The rule fires when
    the votes cover most samples (>= 50% error, >= 10% warning).

    Does NOT apply: time-dependent datasets and effectively 1D fields (skipped as out of scope, see scope_skip); sources that are symmetric under the tested operation (e.g. a single centred hot spot, symmetric
    layouts) give no discrimination and no alarm; non-square grids cannot be tested for transposition (only flips); u
    fields dominated by boundary or k effects rather than by the source (strongly layered k, large convective gradient)
    can show low correlation in every orientation - then the rule stays quiet rather than guessing. When the plain
    correlation prefers another orientation but the band-passed one does not (the source map is dominated by wide
    low-contrast regions, as in some Therm-FM sets) the rule does not fire and reports an info note instead. The k field
    is not tested, but it is normally exported together with the source and shares its orientation.
    """
    res = RuleResult('F003', 'orientation')
    why = scope_skip(ds, opt)
    if why:
        res.status, res.summary = 'skipped', why
        return res
    corr: Dict[str, List[float]] = {}
    bcorr: Dict[str, List[float]] = {}
    better: Dict[str, int] = {}
    n = flagged = raw_only = 0
    ids: List[str] = []
    wide = 4.0 * opt.smooth_sigma
    for i, s in ds.samples(opt.max_samples):
        if s.source is None:
            res.status, res.summary = 'skipped', 'no source field'
            return res
        u, src = np.asarray(s.u, float), np.asarray(s.source, float)
        if not (np.isfinite(u).all() and np.isfinite(src).all()) or np.ptp(src) <= 0 or np.ptp(u) <= 0:
            continue
        sm = gaussian_smooth(src, opt.smooth_sigma)
        c = {nm: _pooled_corr(v, u) for nm, v in _variants(sm).items() if v is not None}
        if not np.isfinite(c['stored']):
            continue
        ub = u - gaussian_smooth(u, wide)
        cb = {nm: _pooled_corr(v, ub) for nm, v in _variants(sm - gaussian_smooth(src, wide)).items() if v is not None}
        n += 1
        for nm in c:
            corr.setdefault(nm, []).append(c[nm])
            bcorr.setdefault(nm, []).append(cb[nm])
        alts = {nm: v for nm, v in c.items() if nm != 'stored' and np.isfinite(v)}
        if alts:
            best = max(alts, key=alts.get)
            if alts[best] > c['stored'] + opt.orientation_margin:
                if cb[best] > cb['stored'] + 0.5 * opt.orientation_margin:
                    flagged += 1
                    better[best] = better.get(best, 0) + 1
                    ids.append(s.id or str(i))
                else:
                    raw_only += 1
    if n < 3:
        res.status, res.summary = 'skipped', f'only {n} usable sample(s)'
        return res
    mean = {nm: float(np.mean(v)) for nm, v in corr.items()}
    bmean = {nm: float(np.nanmean(v)) for nm, v in bcorr.items()}
    frac = flagged / n
    res.metrics = {'checked': n, 'mean_corr': mean, 'mean_corr_bandpass': bmean, 'fraction_alternative_better': frac,
                   'fraction_raw_only_preference': raw_only / n, 'better_by_variant': better,
                   'margin': opt.orientation_margin, 'sigma_cells': opt.smooth_sigma}
    res.summary = 'corr(smoothed source, u): ' + ', '.join(f'{k} {v:.2f}' for k, v in mean.items()) +                   f'; alternative better in {100 * frac:.0f}% of {n} samples'
    if frac >= 0.1:
        res.status = 'flagged'
        top = max(better, key=better.get)
        res.findings.append(Finding('F003', 'error' if frac >= 0.5 else 'warning',
                            f"in {100 * frac:.0f}% of {n} samples the source correlates better with u after '{top}' "
                            f"(mean corr {mean['stored']:.2f} as stored, {mean.get(top, float('nan')):.2f} {top}; band-passed "
                            f"{bmean['stored']:.2f} vs {bmean.get(top, float('nan')):.2f}): the source (and probably k) is "
                            f"stored in a different orientation from u", res.metrics, _short(ids)))
    elif raw_only / n >= 0.5 or mean['stored'] < 0:
        res.findings.append(Finding('F003', 'info', f"the plain smoothed correlation is {mean['stored']:.2f} as stored and "
                            f"favours another orientation in {100 * raw_only / n:.0f}% of samples, but the band-passed correlation "
                            f"({bmean['stored']:.2f} as stored) does not confirm it: large low-contrast source regions "
                            f"dominate the plain measure, so no orientation fault is claimed", res.metrics))
    return res


# ── F004 ─────────────────────────────────────────────────────────────────────────
def _digest(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes() + str(a.shape).encode()).hexdigest()


def f004_duplicates(ds, opt: Options) -> RuleResult:
    """F004 duplicates and train/test leakage.

    Why: the same solution (or the same input) appearing twice inflates the effective sample size, and appearing in
    two splits makes the test metric measure memorisation. u is compared exactly (hash of the stored bytes) and approximately:
    two fields are near-duplicates when max|u1 - u2| <= `dup_rtol` x range (and at least a few float32 ulps), which
    catches copies that went through a float32 round trip or a lossless unit conversion. Candidates are found with a
    (mean, std) bucket lookup and then verified cell by cell. Samples with a split label are compared across splits
    (error) and within a split (warning); inputs (source + k) that repeat across splits while u differs are reported as info.

    Does NOT apply: benchmarks that deliberately repeat a configuration (replicate solves, the same layout under several
    boundary conditions) - expect within-split warnings there; without split labels only within-dataset duplicates can be
    reported, not leakage. Near-duplicates that differ by more than dup_rtol (augmented or slightly perturbed copies) are
    not detected. At most max(max_samples, 2000) samples are compared.
    """
    res = RuleResult('F004', 'duplicates')
    exact: Dict[str, int] = {}
    buckets: Dict[tuple, List[int]] = {}
    store: Dict[int, np.ndarray] = {}
    inputs: Dict[str, List[int]] = {}
    labels: Dict[int, Optional[str]] = {}
    sid: Dict[int, str] = {}
    pairs: List[tuple] = []
    rng_ref = bw_m = bw_s = None
    for i in ds.indices(None if opt.max_samples is None else max(opt.max_samples, 2000)):
        s = ds.get(i)
        if s is None:
            continue
        u = np.asarray(s.u)
        if not np.isfinite(u).all():
            continue
        labels[i], sid[i] = s.split, s.id or str(i)
        if rng_ref is None:
            rng_ref = float(np.ptp(u)) or max(abs(float(u.max())), 1.0)
            bw_m, bw_s = 1e-3 * rng_ref, 1e-3 * rng_ref
        u64 = u.astype(np.float64)
        m, sd = float(u64.mean()), float(u64.std())
        u32 = u64.astype(np.float32).ravel()
        tol = max(opt.dup_rtol * float(np.ptp(u64)), 8 * np.finfo(np.float32).eps * float(np.abs(u64).max()))
        match = None
        h = _digest(u64)
        if h in exact:
            match = exact[h]
        else:
            bm, bs = int(np.floor(m / bw_m)), int(np.floor(sd / bw_s))
            for dm in (-1, 0, 1):
                for ds_ in (-1, 0, 1):
                    for j in buckets.get((bm + dm, bs + ds_), [])[:50]:
                        v = store[j]
                        if v.shape == u32.shape and float(np.max(np.abs(v - u32))) <= tol:
                            match = j
                            break
                    if match is not None:
                        break
                if match is not None:
                    break
            buckets.setdefault((bm, bs), []).append(i)
            store[i] = u32
        exact.setdefault(h, i)
        if match is not None:
            pairs.append((i, match))
        if s.source is not None:
            key = _digest(np.asarray(s.source, np.float64)) + (_digest(np.asarray(s.k, np.float64)) if s.k is not None else '')
            inputs.setdefault(key, []).append(i)
    n = len(labels)
    cross = [(a, b) for a, b in pairs if labels[a] is not None and labels[b] is not None and labels[a] != labels[b]]
    within = [(a, b) for a, b in pairs if (a, b) not in cross]
    in_cross = [v for v in inputs.values() if len(v) > 1 and len({labels[i] for i in v if labels[i] is not None}) > 1]
    res.metrics = {'checked': n, 'duplicate_samples': len(pairs), 'cross_split': len(cross), 'within_split': len(within),
                   'splits_labelled': any(v is not None for v in labels.values()), 'inputs_repeated_across_splits': len(in_cross)}
    if cross:
        res.findings.append(Finding('F004', 'error', f'{len(cross)} sample(s) have a u field identical (to {opt.dup_rtol:g} of '
                            f'its range) to a sample in another split: train/test leakage; e.g. {sid[cross[0][0]]} '
                            f'[{labels[cross[0][0]]}] == {sid[cross[0][1]]} [{labels[cross[0][1]]}]',
                            res.metrics, [sid[a] for a, _ in cross[:5]]))
    if within:
        res.findings.append(Finding('F004', 'warning', f'{len(within)} sample(s) duplicate an earlier sample of the same '
                            f'split (or the dataset has no split labels); e.g. {sid[within[0][0]]} == {sid[within[0][1]]}',
                            res.metrics, [sid[a] for a, _ in within[:5]]))
    if in_cross:
        res.findings.append(Finding('F004', 'info', f'{len(in_cross)} input configuration(s) (source + k) occur in more than '
                            f'one split; fine only if the targets differ for a documented reason (e.g. other boundary conditions)',
                            res.metrics))
    res.status = 'flagged' if any(f.severity != 'info' for f in res.findings) else 'ok'
    res.summary = f'{n} samples compared: {len(cross)} cross-split, {len(within)} within-split duplicate(s)' +                   ('' if res.metrics['splits_labelled'] else ' (no split labels: leakage not testable)')
    return res


# ── F005 ─────────────────────────────────────────────────────────────────────────
def f005_range(ds, opt: Options) -> RuleResult:
    """F005 range and units sanity.

    Why: wrong-unit labels (Celsius stored as kelvin, a rise above ambient stored as absolute temperature), constant
    fields from a failed solve, and non-physical coefficients each train without error. Checks, given the declared units of u
    ('K', 'C', or '' for unspecified/rise): constant (zero-range) fields; for 'K' a minimum below 0 or a maximum below 150
    (looks like Celsius or a rise); for 'C' a minimum below -273.15; any maximum above `max_temperature_k` (silicon melts at
    1687 K, so beyond ~1500 K is a bug, not a hot chip); u below the ambient level when the source is non-negative; k <= 0;
    an all-zero or negative source.

    Does NOT apply: non-thermal fields (declare units '' and only the constant / k / source checks run); cryogenic or
    non-silicon problems (raise or lower the limits via Options); a 'K' dataset that is genuinely a small rise is a false
    alarm on the 150 limit - declare units '' for those.
    """
    res = RuleResult('F005', 'range-units')
    units = (ds.units or '').strip()
    n = const = zero_src = neg_src = bad_k = below_amb = 0
    lo, hi = np.inf, -np.inf
    ids_const: List[str] = []
    for i, s in ds.samples(opt.max_samples):
        u = np.asarray(s.u, float)
        if not np.isfinite(u).all():
            continue
        n += 1
        lo, hi = min(lo, float(u.min())), max(hi, float(u.max()))
        if np.ptp(u) <= 1e-9 * max(1.0, abs(float(u.mean()))):
            const += 1
            ids_const.append(s.id or str(i))
        if s.source is not None:
            src = np.asarray(s.source, float)
            if np.isfinite(src).all():
                zero_src += int(not src.any())
                neg_src += int(src.min() < 0)
                if s.ambient is not None and src.min() >= 0 and u.min() < s.ambient - max(1e-3 * np.ptp(u), 1e-6):
                    below_amb += 1
        if s.k is not None and np.isfinite(s.k).all() and np.min(s.k) <= 0:
            bad_k += 1
    if n == 0:
        res.status, res.summary = 'skipped', 'no finite samples'
        return res
    F = lambda sev, msg, **m: res.findings.append(Finding('F005', sev, msg, {**m, 'min': lo, 'max': hi}))  # noqa: E731
    if const:
        F('error' if const == n else 'warning', f'{const} of {n} sample(s) have a constant u field (failed or unscaled solve)',
          count=const)
        res.findings[-1].samples = _short(ids_const)
    if units.upper() == 'K':
        if lo < 0:
            F('error', f'u declared in K but the minimum is {lo:.4g} (negative absolute temperature)')
        elif hi < 150:
            F('warning', f'u declared in K but the maximum is only {hi:.4g}: looks like Celsius or a rise above ambient')
    elif units.upper() in ('C', 'CELSIUS', 'DEGC'):
        if lo < -273.15:
            F('error', f'u declared in Celsius but the minimum is {lo:.4g} (below absolute zero)')
    kmax = None
    if units.upper() == 'K':
        kmax = opt.max_temperature_k
    elif units.upper() in ('C', 'CELSIUS', 'DEGC'):
        kmax = opt.max_temperature_k - 273.15
    if kmax is not None and hi > kmax:
        F('warning', f'maximum u {hi:.4g} {units} exceeds {kmax:.0f}: beyond silicon melting, likely a unit or solver bug')
    if below_amb:
        F('warning', f'{below_amb} sample(s) have u below the ambient level although the source is non-negative '
                     f'(a passive problem cannot cool below ambient)', count=below_amb)
    if zero_src:
        F('warning', f'{zero_src} sample(s) have an all-zero source', count=zero_src)
    if neg_src:
        F('info', f'{neg_src} sample(s) have negative source values (sinks): the maximum-principle rule F002 skips them',
          count=neg_src)
    if bad_k:
        F('error', f'{bad_k} sample(s) have conductivity k <= 0', count=bad_k)
    res.metrics = {'checked': n, 'u_min': lo, 'u_max': hi, 'units': units or '(unspecified)', 'constant_fields': const}
    res.status = 'flagged' if any(f.severity != 'info' for f in res.findings) else 'ok'
    res.summary = f'u in [{lo:.4g}, {hi:.4g}] ({units or "units unspecified"}); {n} samples; ' + \
                  ('no problems' if not res.findings else f'{len(res.findings)} note(s)')
    return res


# ── F006 ─────────────────────────────────────────────────────────────────────────
def _reduce(a: np.ndarray, max_features: int) -> np.ndarray:
    """Block-average the last two axes until the field has at most max_features entries."""
    st = 1
    while (a.size // (st * st)) > max_features and st < min(a.shape[-2:]):
        st += 1
    if st == 1:
        return a
    y, x = (a.shape[-2] // st) * st, (a.shape[-1] // st) * st
    a = a[..., :y, :x]
    return a.reshape(a.shape[:-2] + (y // st, st, x // st, st)).mean(axis=(-3, -1))


def f006_linearity(ds, opt: Options) -> RuleResult:
    """F006 linearity probe (informational).

    Why: steady conduction is linear in the source for fixed k and boundary conditions, so u = G s with a fixed Green's
    operator G. If sources vary but everything else is fixed, a ridge regression from the (block-averaged) source to u
    recovers G and reaches R^2 near 1; a benchmark in that regime is solved by linear algebra and does not test nonlinear
    surrogates. This rule fits a ridge (penalty chosen on a validation split) on a subsample of at most
    `linear_max_samples` samples with fields block-averaged to <= `linear_max_features` features, and reports held-out R^2
    of the part of u that varies across samples (variance around the training mean field). It never raises severity above
    info.

    Does NOT apply (a low R^2 is not evidence of difficulty): when k, geometry or boundary conditions also vary the map is
    no longer linear in the source alone; with few samples (< 30) or very high-dimensional fields the fit is
    under-determined, so R^2 is a lower bound on what a tuned linear model reaches. It is a screening heuristic, not a
    certificate.
    """
    res = RuleResult('F006', 'linearity-probe')
    X, Y, kvar = [], [], []
    for i, s in ds.samples(opt.linear_max_samples):
        if s.source is None:
            res.status, res.summary = 'skipped', 'no source field'
            return res
        if not (np.isfinite(s.u).all() and np.isfinite(s.source).all()):
            continue
        X.append(_reduce(np.asarray(s.source, float), opt.linear_max_features).ravel())
        Y.append(_reduce(np.asarray(s.u, float), opt.linear_max_features).ravel())
        if s.k is not None:
            kvar.append(_reduce(np.asarray(s.k, float), 64).ravel())
    n = len(X)
    if n < 30:
        res.status, res.summary = 'skipped', f'needs >= 30 samples, have {n}'
        return res
    X, Y = np.array(X), np.array(Y)
    rng = np.random.default_rng(opt.linear_seed)
    perm = rng.permutation(n)
    ntr, nva = int(0.6 * n), int(0.15 * n)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
    mx, my = X[tr].mean(0), Y[tr].mean(0)

    def fit(idx, alpha):
        A, B = X[idx] - mx, Y[idx] - my
        if len(idx) < A.shape[1]:                            # dual form
            return A.T @ np.linalg.solve(A @ A.T + alpha * np.eye(len(idx)), B)
        return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ B)

    def r2(W, idx):
        pred = (X[idx] - mx) @ W + my
        sst = ((Y[idx] - my) ** 2).sum()
        return float(1 - ((Y[idx] - pred) ** 2).sum() / sst) if sst > 0 else float('nan')

    A0 = X[tr] - mx
    scale = float((A0 * A0).sum() / max(len(tr), 1))
    best = max(((r2(fit(tr, c * scale), va), c) for c in (1e-8, 1e-6, 1e-4, 1e-2, 1.0)), key=lambda t: (np.nan_to_num(t[0], nan=-9)))
    W = fit(np.concatenate([tr, va]), best[1] * scale)
    r2_te = r2(W, te)
    k_varies = bool(kvar) and float(np.ptp(np.array(kvar), axis=0).max()) > 1e-9 * max(1.0, float(np.abs(kvar).max()))
    res.metrics = {'samples_used': n, 'features': int(X.shape[1]), 'r2_heldout': r2_te, 'ridge_penalty_rel': best[1],
                   'k_varies': k_varies}
    res.summary = f'ridge from source explains R2 = {r2_te:.3f} of cross-sample variance in u (held out, {len(te)} samples)'
    msg = f'ridge regression from the source field alone explains R^2 = {r2_te:.3f} of the sample-to-sample variance of u ' \
          f'(held-out, n={n}, {X.shape[1]} features)'
    if r2_te >= 0.9:
        msg += ': near-linear benchmark; a nonlinear surrogate must beat this baseline to show value'
    if k_varies:
        msg += ' [k varies across samples, so this is not the linear-in-source regime and R^2 is a floor]'
    res.findings.append(Finding('F006', 'info', msg, res.metrics))
    return res


# ── F007 ─────────────────────────────────────────────────────────────────────────
def f007_energy(ds, opt: Options) -> RuleResult:
    """F007 energy balance (needs boundary flux info).

    Why: in steady state all heat injected by the source leaves through the boundary: flux_out = integral of source dV =
    sum(source) * cell volume. A ratio away from 1 means the solve did not converge, the power field fed to the solver
    differs from the exported one, or the units / cell volume are inconsistent. Only runs when the adapter supplies a
    per-sample `flux_out` (total heat leaving, same units as source x volume); without `spacing` the cell volume is taken
    as 1 (use a unit-consistent flux or supply spacing).

    Does NOT apply: datasets without boundary flux data (the rule is then skipped, never guessed); problems with internal sinks or
    heat storage (transient); fluxes reported per unit area or in different units without a conversion in the adapter
    config; non-uniform grids (a constant cell volume is assumed).
    """
    res = RuleResult('F007', 'energy-balance')
    n = bad = 0
    ratios, ids = [], []
    unit_vol = False
    for i, s in ds.samples(opt.max_samples):
        if s.flux_out is None or s.source is None:
            continue
        dv = float(np.prod(s.spacing)) if s.spacing is not None and all(v is not None for v in s.spacing) else 1.0
        unit_vol |= s.spacing is None
        p_in = float(np.sum(s.source) * dv)
        if abs(p_in) < 1e-300:
            continue
        n += 1
        r = float(s.flux_out) / p_in
        ratios.append(r)
        if abs(r - 1) > opt.energy_tol:
            bad += 1
            ids.append(s.id or str(i))
    if n == 0:
        res.status, res.summary = 'skipped', 'no boundary flux information supplied'
        return res
    res.metrics = {'checked': n, 'violating': bad, 'ratio_min': min(ratios), 'ratio_median': float(np.median(ratios)),
                   'ratio_max': max(ratios), 'tolerance': opt.energy_tol, 'unit_cell_volume_assumed': unit_vol}
    res.summary = f'heat out / heat in median {np.median(ratios):.4f}; {bad}/{n} outside +-{opt.energy_tol:g}'
    if bad:
        res.status = 'flagged'
        res.findings.append(Finding('F007', 'error', f'{bad} of {n} samples have heat out / heat in outside 1 +- {opt.energy_tol:g} '
                            f'(range {min(ratios):.4g} to {max(ratios):.4g})', res.metrics, _short(ids)))
    return res


# ── F008 / F009 operator-residual consistency ────────────────────────────────────
def _orient_variants(a: np.ndarray, shifts: int) -> Dict[str, Optional[np.ndarray]]:
    """Orientations of an input (last two axes) tested by F008: the five of F003 plus circular shifts of 1..`shifts` cells."""
    out = dict(_variants(a))
    for r in range(1, shifts + 1):
        for sgn in (1, -1):
            out[f'roll_x{sgn * r:+d}'] = np.roll(a, sgn * r, axis=-1)
            out[f'roll_y{sgn * r:+d}'] = np.roll(a, sgn * r, axis=-2)
    return out


def _am(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def _gm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))


def _hm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a + b
    return np.where(d > 0, 2.0 * a * b / np.where(d > 0, d, 1.0), 0.0)


def lateral_operator(u: np.ndarray, k: Optional[np.ndarray] = None, dy: float = 1.0, dx: float = 1.0,
                     face: str = 'harmonic') -> np.ndarray:
    """L(u) = -div(k grad u) on the interior cells of a 2D field, conservative 5-point stencil with harmonic-mean face k
    (k = 1 when absent; face = 'arithmetic' or 'geometric' averages the two cell values that way instead). Returns an array of shape (ny-2, nx-2)."""
    c = u[1:-1, 1:-1]
    hm = {'arithmetic': _am, 'geometric': _gm}.get(face, _hm)
    if k is None:
        kE = kW = kN = kS = 1.0
    else:
        kc = k[1:-1, 1:-1]
        kE, kW, kS, kN = hm(kc, k[1:-1, 2:]), hm(kc, k[1:-1, :-2]), hm(kc, k[2:, 1:-1]), hm(kc, k[:-2, 1:-1])
    return -((kE * (u[1:-1, 2:] - c) - kW * (c - u[1:-1, :-2])) / dx ** 2 +
             (kS * (u[2:, 1:-1] - c) - kN * (c - u[:-2, 1:-1])) / dy ** 2)


def _layers(a: Optional[np.ndarray]):
    if a is None:
        return None
    return a[None] if a.ndim == 2 else a


def _layer_modes(src: np.ndarray, k: Optional[np.ndarray]):
    """Per-layer fit mode from the stored inputs: 'var' (source varies), 'uni' (source uniform and non-zero, k varies) or None."""
    modes = []
    for z in range(src.shape[0]):
        s = src[z]
        scale = float(np.abs(s).max())
        if scale > 0 and np.ptp(s) > 1e-6 * scale:
            modes.append('var')
        elif scale > 0 and k is not None and np.ptp(k[z]) > 1e-6 * float(np.abs(k[z]).max()):
            modes.append('uni')
        else:
            modes.append(None)
    return modes


def _op_fit(s: np.ndarray, L: np.ndarray, u: np.ndarray, mode: str):
    """(SSE, SST) of the fit of interior source s. 'var': s ~ a L + b u + d  (d = -b c), SST about the mean of s.
    'uni': s ~ a L, SST = sum s^2 (the source is one constant, so only the operator's shape can be tested)."""
    s, L, u = s.ravel(), L.ravel(), u.ravel()
    if mode == 'uni':
        den = float(L @ L)
        a = float(L @ s) / den if den > 0 else 0.0
        return float(((s - a * L) ** 2).sum()), float((s ** 2).sum())
    # the intercept is eliminated by centring; two scalars remain (normal equations, 2 x 2)
    sc, Lc, uc = s - s.mean(), L - L.mean(), u - u.mean()
    G = np.array([[Lc @ Lc, Lc @ uc], [Lc @ uc, uc @ uc]])
    h = np.array([Lc @ sc, uc @ sc])
    sst = float(sc @ sc)
    d = G[0, 0] * G[1, 1] - G[0, 1] ** 2
    if d > 1e-12 * max(G[0, 0] * G[1, 1], 1e-300):
        coef = np.array([G[1, 1] * h[0] - G[0, 1] * h[1], G[0, 0] * h[1] - G[0, 1] * h[0]]) / d
    else:                                                    # collinear columns: fall back to the better single column
        coef = np.array([h[0] / G[0, 0] if G[0, 0] > 0 else 0.0, 0.0])
    return max(sst - float(coef @ h), 0.0), sst


def operator_r2(src: np.ndarray, k: Optional[np.ndarray], u: np.ndarray, spacing=None, modes=None, ops=None,
                face: str = 'harmonic') -> Optional[float]:
    """R^2 of  s ~ a(-div k grad u) + b(u - c)  pooled over the z-layers of one sample (None if no layer is testable).
    src, k, u are 2D or 3D with matching shape; k may be None. `ops` optionally holds the per-layer L(u) when k is None."""
    S, U, K = _layers(src), _layers(u), _layers(k)
    modes = modes or _layer_modes(S, K)
    dy = dx = 1.0
    if spacing is not None:
        dy, dx = (1.0 if v is None else float(v) for v in spacing[-2:])
    sse = sst = 0.0
    used = 0
    for z, mode in enumerate(modes):
        if mode is None or min(U.shape[-2:]) < 3:
            continue
        L = ops[z] if ops is not None else lateral_operator(U[z], None if K is None else K[z], dy, dx, face)
        e, t = _op_fit(S[z][1:-1, 1:-1], L, U[z][1:-1, 1:-1], mode)
        sse, sst, used = sse + e, sst + t, used + 1
    if used == 0 or sst <= 0:
        return None
    return 1.0 - sse / sst


def _ops(u: np.ndarray, spacing) -> List[np.ndarray]:
    """Per-layer L(u) for k = 1."""
    dy = dx = 1.0
    if spacing is not None:
        dy, dx = (1.0 if v is None else float(v) for v in spacing[-2:])
    return [lateral_operator(l, None, dy, dx) for l in _layers(u)]


def _prep_sample(s):
    """float arrays (source, k, u), or None when the sample cannot be scored."""
    if s.source is None and s.k is None:
        return None
    u = np.asarray(s.u, float)
    src = np.zeros_like(u) if s.source is None else np.asarray(s.source, float)
    k = None if s.k is None else np.asarray(s.k, float)
    if not (np.isfinite(u).all() and np.isfinite(src).all() and (k is None or np.isfinite(k).all())):
        return None
    return src, k, u


def _sample_scores(p, spacing, opt: Options, face: str, modes=None) -> Dict[str, float]:
    """R^2 of the operator fit for each orientation of the inputs of one prepared sample (src, k, u)."""
    src, k, u = p
    modes = modes or _layer_modes(_layers(src), _layers(k))
    sv = _orient_variants(src, opt.op_shifts)
    kv = _orient_variants(k, opt.op_shifts) if k is not None and np.ptp(k) > 0 else None
    ops = _ops(u, spacing) if k is None else None
    out = {}
    for nm, v in sv.items():
        if v is None:
            continue
        r = operator_r2(v, k if kv is None else kv[nm], u, spacing, modes, ops, face)
        if r is not None:
            out[nm] = r
    return out


FACE_MEANS = ('harmonic', 'arithmetic', 'geometric')


def choose_face_mean(ds, opt: Options):
    """(face mean, {mean: median best-orientation R^2}). Fixed by `op_face_mean` unless it is 'auto'.

    'auto' makes the discretisation a fitted choice, one per dataset: on `op_face_samples` evenly spaced samples each face
    mean is scored by the median over samples of the best R^2 over all input orientations (stored and alternatives), and
    the mean with the highest score wins (harmonic on ties and whenever k is absent or constant, where the choice is moot).
    Taking the best over orientations rather than the stored one keeps the choice from being decided by a mis-oriented
    input; it is still decided before, and independently of, the orientation vote."""
    if opt.op_face_mean != 'auto':
        return opt.op_face_mean, {}
    cache = getattr(ds, '_face_cache', None)
    if cache is not None and cache[0] == (opt.op_face_samples, opt.op_shifts):
        return cache[1], cache[2]
    per = {f: [] for f in FACE_MEANS}
    for _i, s in ds.samples(opt.op_face_samples):
        p = _prep_sample(s)
        if p is None or p[1] is None or not np.ptp(p[1]) > 0 or p[2].ndim not in (2, 3) or min(p[2].shape[-2:]) < 3:
            continue
        modes = _layer_modes(_layers(p[0]), _layers(p[1]))
        if not any(modes):
            continue
        for f in FACE_MEANS:
            r2 = _sample_scores(p, s.spacing, opt, f, modes)
            if r2:
                per[f].append(max(r2.values()))
    med = {f: float(np.median(v)) for f, v in per.items() if v}
    face = max(med, key=lambda f: (med[f] > max(med.values()) - 1e-9, f == 'harmonic')) if med else 'harmonic'
    try:
        ds._face_cache = ((opt.op_face_samples, opt.op_shifts), face, med)
    except AttributeError:
        pass
    return face, med


def f008_operator_residual(ds, opt: Options) -> RuleResult:
    """F008 operator-residual consistency.

    Why: a steady diffusion field obeys  -div(k grad u) = s  cell by cell, so the source must be reproduced by the discrete
    operator applied to u. For each sample (each z-layer of a 3D sample) L(u) = -div(k grad u) is formed on the interior
    cells with a conservative 5-point stencil (harmonic-mean face conductivity, k = 1 if absent, the grid spacing if given)
    and s is fitted by least squares as  a L(u) + b (u - c)  with scalars a, b, c (a absorbs units and cell size; b, c model a
    lumped vertical loss to a sink at level c, as in a thin die over a heat sink). The R^2 of that fit is computed with the
    inputs (source and k together) as stored and in every alternative orientation (transposed if square, flipped in x, y,
    both) and, because a one-cell misregistration is also a fault, circularly shifted by 1..`op_shifts` cells along each
    axis. Misaligned inputs make the residual large; the correct alignment makes it small. A sample votes for an
    alternative when its R^2 beats the stored R^2 by `op_margin`; the rule fires when the votes cover >= 10% (warning) / >=
    50% (error) of the samples. When the source is one constant and only k varies (Darcy flow) the fit is s ~ a L(u) with
    no intercept and R^2 measured against sum(s^2): a wrong k orientation makes L(u) non-constant. Unlike F003 this needs no
    smoothing scale and works for k-only problems.

    Does NOT apply (the rule is skipped, with the reason): time-dependent, advective or reaction fields (the operator is not
    the whole equation); fields where neither the source nor k varies; fewer than 3 usable samples; grids under 3 cells; and
    any dataset on which even the best orientation explains a median R^2 below `op_r2_applicable` (the lumped model does not
    describe the data, e.g. a 3D stack with strong inter-layer coupling or an unknown sink, so the comparison would be
    noise). Symmetric inputs (a centred spot, a symmetric k) cannot discriminate orientations. The check treats the inputs
    as exact: a source that is a coarse proxy for the real heating lowers R^2 in every orientation alike. The face
    conductivity is a convention: a dataset generated with a different one (PDEBench Darcy uses the arithmetic mean) is
    reproduced only to the extent the two conventions agree, which for a 10x contrast in k is poorly. The face mean is
    therefore a fitted choice (`op_face_mean = 'auto'`: harmonic, arithmetic or geometric, one per dataset, see
    choose_face_mean) and is reported. Out of scope (skipped): time-dependent datasets and effectively 1D fields (an in-plane
    axis shorter than `min_inplane_cells`).
    """
    res = RuleResult('F008', 'operator-residual')
    why = scope_skip(ds, opt)
    if why:
        res.status, res.summary = 'skipped', why
        return res
    face, face_scores = choose_face_mean(ds, opt)
    scores: Dict[str, List[float]] = {}
    votes: Dict[str, int] = {}
    n = flagged = 0
    ids: List[str] = []
    mode_counts = {'var': 0, 'uni': 0}
    for i, s in ds.samples(opt.max_samples):
        p = _prep_sample(s)
        if p is None:
            continue
        src, k, u = p
        if u.ndim not in (2, 3) or min(u.shape[-2:]) < 3:
            res.status, res.summary = 'skipped', f'u has shape {u.shape}: needs 2D (or z-stacked 2D) fields of at least 3x3 cells'
            return res
        modes = _layer_modes(_layers(src), _layers(k))
        if not any(modes):
            continue
        mode_counts['uni' if all(m in ('uni', None) for m in modes) else 'var'] += 1
        r2 = _sample_scores(p, s.spacing, opt, face, modes)
        if 'stored' not in r2:
            continue
        n += 1
        for nm, r in r2.items():
            scores.setdefault(nm, []).append(r)
        alts = {nm: r for nm, r in r2.items() if nm != 'stored'}
        if alts:
            best = max(alts, key=alts.get)
            if alts[best] > r2['stored'] + opt.op_margin:
                flagged += 1
                votes[best] = votes.get(best, 0) + 1
                ids.append(s.id or str(i))
    if n < opt.op_min_samples:
        res.status, res.summary = 'skipped', ('no sample has a varying source or k (nothing to fit)' if n == 0 else f'only {n} usable sample(s)')
        return res
    med = {nm: float(np.median(v)) for nm, v in scores.items()}
    best_med = max(med.values())
    orient = {k_: v for k_, v in med.items() if not k_.startswith('roll')}
    res.metrics = {'checked': n, 'median_r2': {k_: round(v, 4) for k_, v in med.items()}, 'median_r2_stored': med['stored'],
                   'median_r2_best_orientation': best_med, 'margin': opt.op_margin,
                   'fraction_alternative_better': flagged / n, 'better_by_variant': votes, 'modes': mode_counts,
                   'face_mean': face, 'face_mean_scores': {k_: round(v, 4) for k_, v in face_scores.items()}}
    if best_med < opt.op_r2_applicable:
        res.status = 'skipped'
        res.summary = (f'not applicable: the operator model explains only median R2 = {best_med:.2f} in any orientation '
                       f'(< {opt.op_r2_applicable:g}); the field is probably time-dependent, advective or strongly coupled between layers')
        return res
    frac = flagged / n
    res.summary = f'face mean {face}; median R2 of s ~ aL(u)+b(u-c): ' + ', '.join(f'{k_} {v:.2f}' for k_, v in orient.items()) + \
                  f'; alternative better in {100 * frac:.0f}% of {n} samples'
    if frac >= 0.1:
        res.status = 'flagged'
        top = max(votes, key=votes.get)
        res.findings.append(Finding('F008', 'error' if frac >= 0.5 else 'warning',
                            f"in {100 * frac:.0f}% of {n} samples the inputs reproduce the source through -div(k grad u) better "
                            f"after '{top}' (median R2 {med['stored']:.2f} as stored, {med[top]:.2f} {top}): u, source and k "
                            f"are not aligned as stored", res.metrics, _short(ids)))
    return res


def pair_scores(ds, opt: Options):
    """(items, M): the usable samples and the matrix M[i, j] = operator-fit R^2 of (inputs_i, u_j), stored orientation."""
    face, _ = choose_face_mean(ds, opt)
    items = []
    for i, s in ds.samples(opt.pair_samples):
        p = _prep_sample(s)
        if p is None or not any(_layer_modes(_layers(p[0]), _layers(p[1]))):
            continue
        items.append((i, s, p))
    if len(items) < 4:
        return items, None
    shape = items[0][2][2].shape
    items = [it for it in items if it[2][2].shape == shape]
    m = len(items)
    M = np.full((m, m), np.nan)
    ops = {b: _ops(it[2][2], it[1].spacing) for b, it in enumerate(items)} if all(it[2][1] is None for it in items) else {}
    for a, (_, sa, (src, k, _u)) in enumerate(items):
        modes = _layer_modes(_layers(src), _layers(k))
        for b, (_, _sb, (_s2, _k2, ub)) in enumerate(items):
            r = operator_r2(src, k, ub, sa.spacing, modes, ops.get(b), face)
            if r is not None:
                M[a, b] = r
    return items, M


def f009_pairing(ds, opt: Options) -> RuleResult:
    """F009 sample pairing.

    Why: u_j must be the solution for inputs_j. On M = `pair_samples` evenly spaced samples the operator-fit R^2 of F008
    (stored orientation) is computed for every pair (inputs_i, u_j). When the pairing is right the diagonal (i, i) is the
    best match of row i; a shuffled or off-by-one sample order (inputs from one sample paired with the solution of another)
    puts the best match off the diagonal. A row is 'wrong' when some other column beats the diagonal by more than
    `pair_tie` in R^2. Fires as a warning when >= `pair_warn` and as an error when >= `pair_error` of the rows are wrong.

    Does NOT apply (skipped with the reason): the out-of-scope cases of F008 (time-dependent, effectively 1D); the cases F008 skips (no varying source or k, fields the operator model does not
    describe: the median best-column R^2 is below `pair_r2_applicable`); fewer than 4 usable samples. Datasets whose inputs
    are identical across samples (only boundary conditions vary) have no pairing information and give many ties, which count as
    correct. Samples must have the same grid. A fault that touches only a minority of the samples (< `pair_warn` of the rows)
    is not reported. A pairing fault that destroys the operator fit everywhere makes the rule skip (R^2 below the gate): a
    wrong pairing and an inapplicable operator model both give a diagonal indistinguishable from the off-diagonal entries, so
    the two cannot be told apart from the score matrix alone and the rule does not guess.
    """
    res = RuleResult('F009', 'pairing')
    why = scope_skip(ds, opt)
    if why:
        res.status, res.summary = 'skipped', why
        return res
    items, M = pair_scores(ds, opt)
    face = choose_face_mean(ds, opt)[0]
    if M is None:
        res.status, res.summary = 'skipped', f'only {len(items)} usable sample(s) (needs 4)'
        return res
    m = len(items)
    diag = np.diag(M)
    Mi = np.where(np.isnan(M), -np.inf, M)
    rowmax = Mi.max(axis=1)
    ok = np.isfinite(diag)
    if ok.sum() < 4 or float(np.median(rowmax[ok])) < opt.pair_r2_applicable:
        res.status = 'skipped'
        med = float(np.median(rowmax[ok])) if ok.any() else float('nan')
        res.summary = (f'not applicable: the best match of each row has median R2 {med:.2f} '
                       f'(< {opt.pair_r2_applicable:g}); the operator model does not describe this field')
        return res
    wrong = ok & (rowmax > diag + opt.pair_tie)
    frac = float(wrong.sum() / ok.sum())
    best_col = Mi.argmax(axis=1)
    offs = [int(best_col[a] - a) for a in np.flatnonzero(wrong)]
    consecutive = all(b[0] - a[0] == 1 for a, b in zip(items[:-1], items[1:]))   # the offset is only meaningful for neighbours
    common = max(set(offs), key=offs.count) if offs and consecutive else None
    res.metrics = {'face_mean': face, 'rows': int(ok.sum()), 'wrong_rows': int(wrong.sum()), 'fraction_wrong': frac,
                   'median_diag_r2': float(np.median(diag[ok])), 'median_row_max_r2': float(np.median(rowmax[ok])),
                   'most_common_offset_of_best_partner': common, 'tie': opt.pair_tie}
    res.summary = f"{int(wrong.sum())}/{int(ok.sum())} rows ({100 * frac:.0f}%) match another sample's u better than their own"
    if frac >= opt.pair_warn:
        res.status = 'flagged'
        hint = f'; the best partner is most often {common:+d} positions away' if common not in (None, 0) else ''
        res.findings.append(Finding('F009', 'error' if frac >= opt.pair_error else 'warning',
                            f"for {int(wrong.sum())} of {int(ok.sum())} samples ({100 * frac:.0f}%) the inputs fit another "
                            f"sample's u better than their own (median R2 on the diagonal {np.median(diag[ok]):.2f} vs "
                            f"{np.median(rowmax[ok]):.2f} best){hint}: inputs and targets are probably shuffled or offset",
                            res.metrics, [items[a][1].id or str(items[a][0]) for a in np.flatnonzero(wrong)[:5]]))
    return res


RULES: Dict[str, Callable[..., RuleResult]] = {
    'F001': f001_integrity, 'F002': f002_max_principle, 'F003': f003_orientation, 'F004': f004_duplicates,
    'F005': f005_range, 'F006': f006_linearity, 'F007': f007_energy,
    'F008': f008_operator_residual, 'F009': f009_pairing,
}
