"""Core data model: a Sample is one solved problem; a Dataset is a lazy, indexable collection of them."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Sample:
    """One solved problem. Spatial axes are ordered ([z,] y, x); all spatial fields share u's shape.

    u               output field (2D or 3D), e.g. temperature
    source          optional source field (heat generation), same shape as u
    k               optional coefficient field (conductivity), same shape as u
    spacing         optional grid spacing per spatial axis, same order as the axes of u
    dirichlet_mask  optional boolean field, True where u is held fixed (boundary / ambient nodes)
    flux_out        optional scalar: total heat leaving through the boundary (same units as sum(source)*cell volume)
    ambient         optional boundary / ambient level of u
    """
    u: np.ndarray
    source: Optional[np.ndarray] = None
    k: Optional[np.ndarray] = None
    spacing: Optional[Tuple[float, ...]] = None
    dirichlet_mask: Optional[np.ndarray] = None
    flux_out: Optional[float] = None
    ambient: Optional[float] = None
    id: str = ''
    split: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.u, self.source, self.k, self.dirichlet_mask) if a is not None)


class Dataset:
    """Lazy collection of samples. `loader(i)` builds sample i; failures are recorded, not raised.

    n         number of samples
    loader    callable i -> Sample
    shapes    optional callable () -> dict of field name -> per-sample shape, for cheap shape checks (F001)
    units     declared units of u ('K', 'C', or '' for unknown / dimensionless)
    time_dependent  True when the fields are snapshots of a trajectory; the steady-state rules then report 'out of scope'
    name      label for reports
    """

    def __init__(self, n: int, loader: Callable[[int], Sample], name: str = 'dataset', units: str = '',
                 shapes: Optional[Callable[[], Dict[str, Tuple[int, ...]]]] = None,
                 cache_bytes: int = 800 * 2 ** 20, time_dependent: bool = False):
        self.n, self._loader, self.name, self.units = int(n), loader, name, units
        self.time_dependent = bool(time_dependent)
        self.field_shapes = shapes
        self.errors: Dict[int, str] = {}
        self._cache: 'OrderedDict[int, Optional[Sample]]' = OrderedDict()
        self._cache_bytes, self._cached = cache_bytes, 0
        self.notes: List[str] = []

    def __len__(self):
        return self.n

    def get(self, i: int) -> Optional[Sample]:
        """Sample i, or None if it could not be loaded (the reason is in self.errors[i])."""
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        try:
            s = self._loader(i)
        except Exception as exc:                                # noqa: BLE001 - reported by F001
            self.errors[i] = f'{type(exc).__name__}: {exc}'
            s = None
        self._cache[i] = s
        self._cached += s.nbytes() if s is not None else 0
        while self._cached > self._cache_bytes and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cached -= old.nbytes() if old is not None else 0
        return s

    def close(self):
        """Release open file handles (HDF5 files stay locked on Windows until closed)."""
        for f in (getattr(self, '_keep', None) or {}).values():
            f.close()

    def head(self, n: int) -> 'Dataset':
        """A view on the first n samples (split labels are unchanged)."""
        d = Dataset(min(n, self.n), self._loader, self.name, self.units, self.field_shapes, time_dependent=self.time_dependent)
        d._cache, d._keep = self._cache, getattr(self, '_keep', None)
        return d

    def indices(self, max_samples: Optional[int] = None) -> List[int]:
        """All indices, or `max_samples` evenly spaced ones (deterministic)."""
        if not max_samples or max_samples >= self.n:
            return list(range(self.n))
        return sorted(set(np.linspace(0, self.n - 1, max_samples).round().astype(int).tolist()))

    def samples(self, max_samples: Optional[int] = None):
        for i in self.indices(max_samples):
            s = self.get(i)
            if s is not None:
                yield i, s


def dataset_from_arrays(u, source=None, k=None, *, spacing=None, dirichlet_mask=None, flux_out=None, ambient=None,
                        splits: Optional[Sequence[Optional[str]]] = None, units: str = '', name: str = 'arrays') -> Dataset:
    """Build a Dataset from in-memory stacks of shape (N, [Z,] Y, X). Convenient for tests and notebooks.

    dirichlet_mask may be one mask shared by all samples (shape of u[0]) or a stack.
    """
    u = np.asarray(u)

    def pick(a, i):
        return None if a is None else np.asarray(a)[i]

    def mask(i):
        if dirichlet_mask is None:
            return None
        m = np.asarray(dirichlet_mask)
        return m[i] if m.ndim == u.ndim else m

    def load(i):
        return Sample(u=u[i], source=pick(source, i), k=pick(k, i), spacing=spacing, dirichlet_mask=mask(i),
                      flux_out=None if flux_out is None else float(np.asarray(flux_out)[i]), ambient=ambient,
                      id=f'{name}[{i}]', split=None if splits is None else splits[i])

    shapes = {'u': tuple(u.shape[1:])}
    for nm, a in (('source', source), ('k', k)):
        if a is not None:
            shapes[nm] = tuple(np.asarray(a).shape[1:])
    return Dataset(len(u), load, name=name, units=units, shapes=lambda: shapes)
