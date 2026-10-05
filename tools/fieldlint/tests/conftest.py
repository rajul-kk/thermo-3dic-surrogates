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
