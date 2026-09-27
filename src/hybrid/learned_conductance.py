"""Learn the physics, then solve it exactly (report §9.31).

A network maps the power-free inputs (layout, material, cooling) to a local 2D conductance field: per-cell lateral
conductivity kappa and per-cell conductance to ambient g. Temperature rise is then the exact solution of the discrete
conservation law  A(kappa, g) theta = q, with A the 5-point finite-volume operator (harmonic-mean face conductances,
adiabatic edges, sink g in every cell). Every prediction is therefore a genuine steady-state solution: exactly linear in
power, conservative, and determined by a *local* map from materials to conductances, which is what should transfer to
unseen packages. The solve is preconditioned CG (preconditioner: the DCT-diagonal operator with the sample's mean kappa
and g, the 2D analogue of the §9.25 backbone) and is differentiated implicitly: one adjoint solve per backward pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_matrix(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    m = torch.cos(math.pi * (k[None, :] + 0.5) * k[:, None] / n)
    m[0] /= math.sqrt(2.0)
    return (m * math.sqrt(2.0 / n)).to(dtype=dtype, device=device)


def face_conductances(kappa):
    """Harmonic-mean conductances on x faces (B, nx-1, ny) and y faces (B, nx, ny-1)."""
    gx = 2 * kappa[:, :-1] * kappa[:, 1:] / (kappa[:, :-1] + kappa[:, 1:])
    gy = 2 * kappa[:, :, :-1] * kappa[:, :, 1:] / (kappa[:, :, :-1] + kappa[:, :, 1:])
    return gx, gy


def apply_A(x, kappa, g):
    """A x for the 5-point FV operator with adiabatic edges and per-cell sink g. x, kappa, g: (B, nx, ny)."""
    gx, gy = face_conductances(kappa)
    y = g * x
    fx = gx * (x[:, :-1] - x[:, 1:])
    fy = gy * (x[:, :, :-1] - x[:, :, 1:])
    y = y + F.pad(fx, (0, 0, 0, 1)) - F.pad(fx, (0, 0, 1, 0))
    y = y + F.pad(fy, (0, 1)) - F.pad(fy, (1, 0))
    return y


class _Precond:
    """Exact inverse of the operator with the sample's mean kappa and mean g (diagonal in the 2D DCT basis)."""

    def __init__(self, kappa, g):
        B, nx, ny = kappa.shape
        dev, dt = kappa.device, kappa.dtype
        self.Dx, self.Dy = dct_matrix(nx, dt, dev), dct_matrix(ny, dt, dev)
        ex = (2 * torch.sin(math.pi * torch.arange(nx, device=dev, dtype=dt) / (2 * nx))) ** 2
        ey = (2 * torch.sin(math.pi * torch.arange(ny, device=dev, dtype=dt) / (2 * ny))) ** 2
        kb = kappa.mean((1, 2))[:, None, None]
        gb = g.mean((1, 2))[:, None, None]
        self.inv = 1.0 / (kb * (ex[None, :, None] + ey[None, None, :]) + gb)

    def __call__(self, r):
        rh = torch.einsum('px,bxy,qy->bpq', self.Dx, r, self.Dy)
        return torch.einsum('px,bpq,qy->bxy', self.Dx, rh * self.inv, self.Dy)


def pcg(b, kappa, g, tol=1e-7, max_iter=500):
    """Batched preconditioned CG for A(kappa, g) x = b. Returns (x, iterations used, worst relative residual)."""
    M = _Precond(kappa, g)
    x = torch.zeros_like(b)
    r = b.clone()
    z = M(r)
    p = z.clone()
    rz = (r * z).sum((1, 2))
    bn = b.flatten(1).norm(dim=1).clamp_min(1e-30)
    it, rel = 0, torch.ones(())
    for it in range(1, max_iter + 1):
        Ap = apply_A(p, kappa, g)
        alpha = rz / (p * Ap).sum((1, 2)).clamp_min(1e-30)
        x = x + alpha[:, None, None] * p
        r = r - alpha[:, None, None] * Ap
        rel = (r.flatten(1).norm(dim=1) / bn).max()
        if rel < tol:
            break
        z = M(r)
        rz_new = (r * z).sum((1, 2))
        p = z + (rz_new / rz.clamp_min(1e-30))[:, None, None] * p
        rz = rz_new
    return x, it, float(rel)


class ConductanceSolve(torch.autograd.Function):
    """theta = A(kappa, g)^-1 q, with implicit (adjoint) gradients; A is symmetric, so the adjoint is the same solve."""

    @staticmethod
    def forward(ctx, q, kappa, g, tol, max_iter):
        with torch.no_grad():
            x, it, rel = pcg(q.double(), kappa.double(), g.double(), tol, max_iter)
        ctx.save_for_backward(x, kappa, g)
        ctx.tol, ctx.max_iter = tol, max_iter
        ctx.stats = (it, rel)
        return x.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_x):
        x, kappa, g = ctx.saved_tensors
        with torch.no_grad():
            lam, _, _ = pcg(grad_x.double(), kappa.double(), g.double(), ctx.tol, ctx.max_iter)
        with torch.enable_grad():
            k = kappa.detach().double().requires_grad_(True)
            gg = g.detach().double().requires_grad_(True)
            Ax = apply_A(x, k, gg)
            gk, gg_ = torch.autograd.grad(Ax, (k, gg), grad_outputs=-lam)
        return lam.to(grad_x.dtype), gk.to(kappa.dtype), gg_.to(g.dtype), None, None


class _Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, padding_mode='replicate'), nn.GroupNorm(min(8, co), co),
                                 nn.GELU(), nn.Conv2d(co, co, 3, padding=1, padding_mode='replicate'), nn.GELU())

    def forward(self, x):
        return self.net(x)


class ConductanceNet(nn.Module):
    """Small U-Net: power-free input maps (B, G, nx, ny) -> log kappa, log g (B, nx, ny). Never sees power."""

    def __init__(self, geo_in: int, ch: int = 32, levels: int = 3, kappa0: float = 1.0, g0: float = 6e-3):
        super().__init__()
        self.inc = _Block(geo_in, ch)
        self.down = nn.ModuleList(_Block(ch * 2 ** i, ch * 2 ** (i + 1)) for i in range(levels))
        self.up = nn.ModuleList(_Block(ch * 2 ** (i + 1) + ch * 2 ** i, ch * 2 ** i) for i in reversed(range(levels)))
        self.head = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([math.log(kappa0), math.log(g0)]))

    def forward(self, geo):
        x = self.inc(geo)
        skips = [x]
        for d in self.down:
            x = d(F.avg_pool2d(x, 2, ceil_mode=True))
            skips.append(x)
        skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False), s], 1))
        out = self.head(x).clamp(-20, 20)
        return out[:, 0].exp(), out[:, 1].exp()


class LearnedConductanceSolver(nn.Module):
    """T = T_amb + b + A(kappa(geo), g(geo))^-1 q. Exactly linear in q for fixed inputs; b is one learned offset."""

    def __init__(self, geo_in: int, ch: int = 32, levels: int = 3, tol: float = 1e-6, max_iter: int = 400):
        super().__init__()
        self.net = ConductanceNet(geo_in, ch, levels)
        self.b = nn.Parameter(torch.zeros(()))
        self.tol, self.max_iter = tol, max_iter

    def fields(self, geo):
        return self.net(geo[..., 0])                       # drop the singleton z axis

    def forward(self, q, geo, amb):
        kappa, g = self.fields(geo)
        theta = ConductanceSolve.apply(q[:, 0, ..., 0], kappa, g, self.tol, self.max_iter)
        return (amb[:, None, None] + self.b + theta)[..., None]
