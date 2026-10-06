"""Synthetic, physically valid samples: a 2D Poisson/heat problem  -lap(u) = s  with u = 0 on the exterior, solved with numpy."""
import numpy as np
import pytest

AMBIENT = 300.0


def poisson_matrix(n):
    """Dense 5-point Laplacian on an n x n grid with homogeneous Dirichlet (u = 0 just outside the grid)."""
    idx = np.arange(n * n).reshape(n, n)
    A = 4.0 * np.eye(n * n)
    for a, b in ((idx[1:], idx[:-1]), (idx[:, 1:], idx[:, :-1])):
        A[a.ravel(), b.ravel()] = -1.0
        A[b.ravel(), a.ravel()] = -1.0
    return A


def random_blocks(rng, n_samples, n, kmax=3):
    """Non-negative source maps made of 1..kmax random rectangles (not symmetric under transpose or flips)."""
    s = np.zeros((n_samples, n, n))
    for i in range(n_samples):
        for _ in range(rng.integers(1, kmax + 1)):
            h, w = rng.integers(2, 5), rng.integers(2, 7)
            y, x = rng.integers(1, n - h - 1), rng.integers(1, n - w - 1)
            s[i, y:y + h, x:x + w] += rng.uniform(0.5, 2.0)
    return s


def solve(src, scale=20.0):
    """Temperature field (K) for each source map; peak rise over the dataset is about `scale` kelvin."""
    n = src.shape[-1]
    u = np.linalg.solve(poisson_matrix(n), src.reshape(len(src), -1).T).T.reshape(src.shape)
    return AMBIENT + u * (scale / u.max())


@pytest.fixture(scope='session')
def valid():
    """(source, temperature) stacks, N=48, 24 x 24, physically valid."""
    rng = np.random.default_rng(0)
    src = random_blocks(rng, 48, 24)
    return src, solve(src)


def solve_k(k, beta=1.0, face='harmonic'):
    """-div(k grad u) = beta on an n x n grid, u = 0 just outside, harmonic-mean face k (boundary faces use the cell's own k).
    k: (N, n, n). Returns u (N, n, n)."""
    N, n, _ = k.shape
    out = np.empty_like(k, dtype=float)
    idx = np.arange(n * n).reshape(n, n)
    for m in range(N):
        kk = k[m]
        A = np.zeros((n * n, n * n))
        for y in range(n):
            for x in range(n):
                i = idx[y, x]
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < n and 0 <= xx < n:
                        kf = 2 * kk[y, x] * kk[yy, xx] / (kk[y, x] + kk[yy, xx]) if face == 'harmonic' else 0.5 * (kk[y, x] + kk[yy, xx])
                        A[i, idx[yy, xx]] -= kf
                    else:
                        kf = kk[y, x]
                    A[i, i] += kf
        out[m] = np.linalg.solve(A, np.full(n * n, beta)).reshape(n, n)
    return out


@pytest.fixture(scope='session')
def darcy_like():
    """k-only problem: uniform source, piecewise-constant random k (two levels, asymmetric blobs), N=24, 20 x 20."""
    rng = np.random.default_rng(1)
    n = 20
    k = np.ones((24, n, n))
    for m in range(24):
        for _ in range(rng.integers(2, 5)):
            h, w = rng.integers(3, 8), rng.integers(3, 11)
            y, x = rng.integers(0, n - h), rng.integers(0, n - w)
            k[m, y:y + h, x:x + w] = 0.1
    return k, solve_k(k, 1.0)
