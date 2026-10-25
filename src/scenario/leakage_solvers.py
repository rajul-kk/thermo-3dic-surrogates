"""Classical full-field solvers for per-block leakage feedback (docs/protocol_nonlinear_A.md section 5, amendment 15).

Plant: the 9.23 finite-volume conduction operator A (same cells as 3D-ICE), block power maps E (n x B, W per cell per W/cm^2),
block-mean operator W (B x n) on the source cell layer. Temperature rise theta = A^-1 E d, T_b = T_amb + W theta.
Methods (all use the layered DCT x tridiagonal backbone of 9.30 as the PCG preconditioner, no training):
  C0  linearised superposition: nominal solve, evaluate P(T_nom), one more solve (2 solves).
  C1  damped Picard on the block powers with warm-started inner PCG solves.
  C2  Newton-PCG on the full field: F(theta) = A theta - E P(W theta), J = A - E D W. Because E = W^T diag(area_b) for cell-aligned
      blocks, J is symmetric, and SPD whenever the loop gain is < 1, so the backbone preconditions it directly.
Each method returns the iterate after every step with the cumulative wall time of the solver work, so error-vs-time curves can be
read off by truncation. Powers are densities in W/cm^2 (the pipeline unit); temperatures in C unless stated.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp

from src.scenario.leakage_perblock import block_power, dlaw_dt, law_arrhenius, law_exp

LAWS = {'exp': law_exp, 'arrhenius': law_arrhenius}
TREF = 85.0


def pcg(op: Callable, M: Callable, b: np.ndarray, x0: Optional[np.ndarray] = None, rtol: float = 1e-6, maxit: int = 500):
    """Preconditioned CG for op x = b. Returns (x, iterations, ok); ok is False if the operator was not positive definite."""
    nb = float(np.linalg.norm(b))
    x = np.zeros_like(b) if x0 is None else x0.copy()
    if nb == 0.0:
        return np.zeros_like(b), 0, True
    r = b - op(x) if x0 is not None else b.copy()
    if np.linalg.norm(r) <= rtol * nb:
        return x, 0, True
    z = M(r); p = z.copy(); rz = float(r @ z)
    for it in range(1, maxit + 1):
        Ap = op(p); pAp = float(p @ Ap)
        if pAp <= 0.0:
            return x, it, False
        a = rz / pAp; x += a * p; r -= a * Ap
        if np.linalg.norm(r) <= rtol * nb:
            return x, it, True
        z = M(r); rz_new = float(r @ z); p = z + (rz_new / rz) * p; rz = rz_new
    return x, maxit, True


class Plant:
    """geometry1 conduction plant at one HTC. Built once per HTC; per-scenario work is only solves."""

    def __init__(self, htc: float, geometry_name: str = 'geometry1'):
        from scripts.backbone_pcg import backbone_op
        from src.core.geometry_builders import get_geometry_by_name
        from src.validation import fv_solver as fv
        t0 = time.perf_counter()
        self.htc, self.geometry, self.fv = float(htc), get_geometry_by_name(geometry_name), fv
        g = self.g = fv.make_grid(self.geometry)
        scen = {'power_blocks': {}, 'htc': self.htc, 't_ambient': 25.0, 'layer_k_overrides': {}}
        kl, kv = fv.conductivity(self.geometry, scen, g)
        self.A, _ = fv.assemble(g, kl, kv, self.htc)
        self.M = backbone_op(g, kl, kv, self.htc)
        self.names = [b.name for b in self.geometry.power_blocks]
        self.n = int(np.prod(g.shape))
        cols = []
        for name in self.names:
            sc = dict(scen, power_blocks={nm: (1.0 if nm == name else 0.0) for nm in self.names})
            cols.append(fv.power(self.geometry, sc, g).ravel())
        self.E = np.array(cols).T                                            # n x B, W per cell per W/cm^2
        z = np.nonzero(fv.source_cells(self.geometry, g))[0]
        if len(z) != 1:
            raise RuntimeError(f'expected one source cell layer, got {z}')
        self.zsrc = int(z[0])
        nx, ny, nz = g.shape
        rows, cidx, vals = [], [], []
        for bi, blk in enumerate(self.geometry.power_blocks):
            m = fv._centre_mask(g, blk.x, blk.y, blk.width, blk.height)
            ij = np.argwhere(m)
            flat = (ij[:, 0] * ny + ij[:, 1]) * nz + self.zsrc
            rows += [bi] * len(flat); cidx += flat.tolist(); vals += [1.0 / len(flat)] * len(flat)
        self.W = sp.csr_matrix((vals, (rows, cidx)), shape=(len(self.names), self.n))
        self.area_cm2 = np.array([b.width * b.height * 1e-8 for b in self.geometry.power_blocks])
        self.setup_s = time.perf_counter() - t0
        # unit responses to machine tolerance: X (n x B) and R = W X (K per W/cm^2); also the amortised cost of this set-up
        t0 = time.perf_counter(); self.X = np.zeros_like(self.E); self.unit_iters = []
        for j in range(self.E.shape[1]):
            x, it, _ = pcg(self.A.dot, self.M, self.E[:, j], rtol=1e-11, maxit=1000)
            self.X[:, j] = x; self.unit_iters.append(it)
        self.R = self.W @ self.X
        self.unit_s = time.perf_counter() - t0

    def block_T(self, theta: np.ndarray, t_amb: float) -> np.ndarray:
        return t_amb + self.W @ theta

    def symmetry_defect(self) -> float:
        """max |E - W^T diag(area)| / max|E|: zero when the Newton Jacobian A - E D W is symmetric."""
        return float(np.abs(self.E - self.W.T.toarray() * self.area_cm2[None, :]).max() / np.abs(self.E).max())


@dataclass
class Problem:
    """One leakage scenario: nominal = dynamic + leakage at Tref, per block, in W/cm^2."""
    t_amb: float
    nominal: np.ndarray
    share: np.ndarray
    k_d: float
    law: str = 'exp'

    @property
    def p_dyn(self): return self.nominal * (1.0 - self.share)

    @property
    def p_leak(self): return self.nominal * self.share

    def power(self, T_b):
        return block_power(T_b, self.p_dyn, self.p_leak, TREF, self.k_d, LAWS[self.law])

    def dpdt(self, T_b):
        return self.p_leak * dlaw_dt(LAWS[self.law], T_b, TREF, self.k_d)


@dataclass
class Run:
    method: str
    states: List[np.ndarray] = field(default_factory=list)      # theta (K rise) after each step
    times: List[float] = field(default_factory=list)            # cumulative solver wall time, s
    inner: List[int] = field(default_factory=list)              # cumulative PCG iterations
    resid: List[float] = field(default_factory=list)            # method's own residual after each step
    converged: bool = False
    runaway: bool = False

    def push(self, theta, t, inner, resid):
        self.states.append(theta); self.times.append(t); self.inner.append(inner); self.resid.append(resid)


def _solve_power(plant: Plant, d, x0, rtol):
    return pcg(plant.A.dot, plant.M, plant.E @ d, x0, rtol)


def c0(plant: Plant, pb: Problem, rtol: float = 1e-6) -> Run:
    run = Run('C0'); t0 = time.perf_counter(); inner = 0
    th, it, _ = _solve_power(plant, pb.nominal, None, rtol); inner += it
    run.push(th, time.perf_counter() - t0, inner, float('nan'))
    d1 = pb.power(plant.block_T(th, pb.t_amb))
    th1, it, _ = _solve_power(plant, d1, th, rtol); inner += it
    run.push(th1, time.perf_counter() - t0, inner, float('nan'))
    return run


def c1(plant: Plant, pb: Problem, w: float = 0.5, tol: float = 1e-6, max_iter: int = 200, rtol: float = 1e-6, t_runaway: float = 300.0) -> Run:
    """Picard of protocol section 3 (start at nominal power, w -> 1 once the gain estimate is < 0.5), inner PCG warm-started."""
    run = Run('C1'); t0 = time.perf_counter(); inner = 0
    d = pb.nominal.copy(); th = None; prev = None; grow = 0
    for k in range(1, max_iter + 1):
        th, it, _ = _solve_power(plant, d, th, rtol); inner += it
        T = plant.block_T(th, pb.t_amb); dn = pb.power(T)
        res = float(np.max(np.abs(dn - d) / d))
        run.push(th, time.perf_counter() - t0, inner, res)
        if T.max() > t_runaway: run.runaway = True; return run
        if res < tol: run.converged = True; return run
        if prev is not None:
            grow = grow + 1 if res > prev else 0
            if grow >= 3: run.runaway = True; return run
            if res / prev < 0.5: w = 1.0
        prev = res
        d = (1.0 - w) * d + w * dn
    return run


def c2(plant: Plant, pb: Problem, tol: float = 1e-8, max_iter: int = 30, rtol_nom: float = 1e-6, eta: float = 1e-4, t_runaway: float = 300.0) -> Run:
    """Newton-PCG on the full field. Start: the nominal-power solve. Jacobian-vector product needs no extra solve."""
    run = Run('C2'); t0 = time.perf_counter(); inner = 0
    th, it, _ = _solve_power(plant, pb.nominal, None, rtol_nom); inner += it
    for k in range(max_iter):
        T = plant.block_T(th, pb.t_amb)
        if T.max() > t_runaway: run.runaway = True; run.push(th, time.perf_counter() - t0, inner, float('nan')); return run
        P = pb.power(T); rhs_nl = plant.E @ P
        F = plant.A.dot(th) - rhs_nl; rel = float(np.linalg.norm(F) / np.linalg.norm(rhs_nl))
        run.push(th, time.perf_counter() - t0, inner, rel)
        if rel < tol: run.converged = True; return run
        D = pb.dpdt(T)
        op = lambda v: plant.A.dot(v) - plant.E @ (D * (plant.W @ v))
        dlt, it, ok = pcg(op, plant.M, -F, None, eta); inner += it
        if not ok: run.runaway = True; return run
        th = th + dlt
    return run


def reference(plant: Plant, pb: Problem):
    """Exact fixed point of the FV plant (4x4 reduction with machine-precision unit responses): (d*, theta*, T_b*) or None (runaway)."""
    dyn, lk = pb.p_dyn, pb.p_leak
    d = dyn.copy()
    for _ in range(100000):
        T = pb.t_amb + plant.R @ d
        if T.max() > 300.0: return None
        dn = pb.power(T)
        if np.max(np.abs(dn - d) / dn) < 1e-13:
            return dn, plant.X @ dn, pb.t_amb + plant.R @ dn
        d = dn
    return None


def loop_gain(plant: Plant, pb: Problem, T_b) -> float:
    """Spectral radius of diag(dP/dT) R at block temperatures T_b; < 1 is the stability (Newton-PCG SPD) condition."""
    return float(np.max(np.abs(np.linalg.eigvals(np.diag(pb.dpdt(T_b)) @ plant.R))))


def classify(plant: Plant, pb: Problem, max_iter: int = 20000):
    """Stability test on the 4x4 reduction: monotone iteration from the dynamic power; runaway as soon as the loop gain reaches 1 or
    T exceeds 300 C. Returns (predicted_converged, gain_at_exit, steps)."""
    d = pb.p_dyn.copy()
    for k in range(max_iter):
        T = pb.t_amb + plant.R @ d
        rho = loop_gain(plant, pb, T) if T.max() < 300.0 else np.inf
        if rho >= 1.0: return False, float(rho), k
        dn = pb.power(T)
        if np.max(np.abs(dn - d) / dn) < 1e-12: return True, float(rho), k
        d = dn
    return False, float('nan'), max_iter
