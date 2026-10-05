"""Rules F001-F007.

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

    Does NOT apply: problems with sinks (s < 0, e.g. Peltier cooling or radiation modelled as a source) - samples with
    negative source are skipped; time-dependent or advection-dominated fields; fields where every cell is powered (the
    check is vacuous and reported as such); datasets whose source is a coarse proxy of the true heating (e.g. a
    block-averaged power map on a finer grid) - raise `neighbourhood`.
    """
    res = RuleResult('F002', 'max-principle')
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

    Does NOT apply: sources that are symmetric under the tested operation (e.g. a single centred hot spot, symmetric
    layouts) give no discrimination and no alarm; non-square grids cannot be tested for transposition (only flips); u
    fields dominated by boundary or k effects rather than by the source (strongly layered k, large convective gradient)
    can show low correlation in every orientation - then the rule stays quiet rather than guessing. When the plain
    correlation prefers another orientation but the band-passed one does not (the source map is dominated by wide
    low-contrast regions, as in some Therm-FM sets) the rule does not fire and reports an info note instead. The k field
    is not tested, but it is normally exported together with the source and shares its orientation.
    """
    res = RuleResult('F003', 'orientation')
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


RULES: Dict[str, Callable[..., RuleResult]] = {
    'F001': f001_integrity, 'F002': f002_max_principle, 'F003': f003_orientation, 'F004': f004_duplicates,
    'F005': f005_range, 'F006': f006_linearity, 'F007': f007_energy,
}
