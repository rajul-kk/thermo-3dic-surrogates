"""Adapters: turn files on disk into a lazy fieldlint Dataset, driven by a small mapping config.

Config (dict from JSON/YAML, a preset, or CLI flags):

    {
      "format": "auto",                  # auto | npz | npy | mat | h5 | 3dice
      "units": "K", "ambient": null, "spacing": [1.0, 1.0],
      "u":      {"key": "temp",  "layout": "NHW"},
      "source": {"key": "power", "layout": "NHW"},
      "k":      {"key": "kmap",  "layout": "NHW", "optional": true},
      "dirichlet_mask": {...}, "flux_out": {"key": "q", "layout": "N"},
      "splits": "none" | "from_filename" | {"rule": "ictherm"} | {"train": [0, 0.8], "test": [0.8, 1]}
    }

Each field spec may also carry "file" (relative to the dataset directory; default = the dataset file itself) and "channel"
(index along the C axis). Layout letters describe the array exactly as the reader returns it: N sample, C channel,
Z/D/L depth, H/Y rows, W/X columns. Omit N for files that hold one sample each (a directory of such files is then one sample
per file). For MATLAB v7.3 files the layout is in h5py order, which is the reverse of MATLAB's order.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Dataset, Sample

FIELDS = ('u', 'source', 'k', 'dirichlet_mask', 'flux_out')
_LETTER = {'N': 'N', 'C': 'C', 'Z': 'Z', 'D': 'Z', 'L': 'Z', 'H': 'Y', 'Y': 'Y', 'W': 'X', 'X': 'X'}
PRESET_DIR = Path(__file__).parent / 'presets'


class AdapterError(ValueError):
    pass


def norm_layout(layout: str) -> str:
    try:
        out = ''.join(_LETTER[c] for c in layout.upper())
    except KeyError as exc:
        raise AdapterError(f"bad layout letter {exc} in '{layout}' (use N, C, Z/D/L, H/Y, W/X)") from None
    if len(set(out)) != len(out) or 'Y' not in out or 'X' not in out:
        raise AdapterError(f"layout '{layout}' must have unique letters and include H and W")
    return out


# ── config loading ───────────────────────────────────────────────────────────────
def load_config(path_or_name: str) -> Dict[str, Any]:
    """A preset name (ictherm, thermfm, 3dice), or a path to a .json / .yaml / .yml file."""
    p = Path(path_or_name)
    if not p.exists():
        preset = PRESET_DIR / f'{path_or_name}.json'
        if preset.exists():
            p = preset
        else:
            raise AdapterError(f"no such config file or preset: {path_or_name} (presets: {', '.join(list_presets())})")
    text = p.read_text(encoding='utf-8')
    if p.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError:
            raise AdapterError('YAML configs need pyyaml: pip install fieldlint[yaml]') from None
        return yaml.safe_load(text) or {}
    return json.loads(text)


def list_presets() -> List[str]:
    return sorted(p.stem for p in PRESET_DIR.glob('*.json'))


# ── array readers ────────────────────────────────────────────────────────────────
class _File:
    """One data file; arrays are fetched lazily by key. npz/mat v5 are held in memory, npy is memory-mapped, HDF5 is lazy."""

    def __init__(self, path: Path):
        self.path, self._h5, self._arrays = path, None, {}
        ext = path.suffix.lower()
        self.kind = {'.npz': 'npz', '.npy': 'npy', '.mat': 'mat', '.h5': 'h5', '.hdf5': 'h5', '.hdf': 'h5'}.get(ext)
        if self.kind is None:
            raise AdapterError(f'unsupported file type: {path}')
        if self.kind == 'npz':
            self._npz = np.load(path, allow_pickle=False)
        elif self.kind == 'mat':
            try:
                import scipy.io as sio
            except ImportError:
                sio = None
            self._mat = None
            if sio is not None:
                try:
                    self._mat = sio.loadmat(str(path), squeeze_me=False)
                except NotImplementedError:
                    self._mat = None                         # v7.3 -> HDF5
                except Exception as exc:                     # noqa: BLE001
                    if 'Unknown mat file type' in str(exc):  # scipy's message for MATLAB v7.3 (HDF5) files
                        self._mat = None
                    else:
                        raise AdapterError(f'cannot read {path} with scipy: {exc}') from None
            if self._mat is None:
                self._h5 = self._open_h5(path, 'MATLAB v7.3 files need h5py: pip install fieldlint[h5]')
        elif self.kind == 'h5':
            self._h5 = self._open_h5(path, 'HDF5 files need h5py: pip install fieldlint[h5]')

    @staticmethod
    def _open_h5(path, msg):
        try:
            import h5py
        except ImportError:
            raise AdapterError(msg) from None
        return h5py.File(path, 'r')

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._arrays.clear()

    def keys(self) -> List[str]:
        if self.kind == 'npz':
            return list(self._npz.files)
        if self.kind == 'npy':
            return ['']
        if self.kind == 'mat' and self._mat is not None:
            return [k for k in self._mat if not k.startswith('__')]
        names: List[str] = []
        self._h5.visititems(lambda n, o: names.append(n) if hasattr(o, 'shape') else None)
        return names

    def get(self, key: Optional[str]):
        k = key or ''
        if k in self._arrays:
            return self._arrays[k]
        if self.kind == 'npy':
            a = np.load(self.path, mmap_mode='r')
        elif self.kind == 'npz':
            if k not in self._npz.files:
                raise AdapterError(f"key '{k}' not in {self.path.name}; keys: {self.keys()}")
            a = self._npz[k]
        elif self.kind == 'mat' and self._mat is not None:
            if k not in self._mat:
                raise AdapterError(f"key '{k}' not in {self.path.name}; keys: {self.keys()}")
            a = self._mat[k]
        else:
            if k not in self._h5:
                raise AdapterError(f"key '{k}' not in {self.path.name}; keys: {self.keys()}")
            a = self._h5[k]
        self._arrays[k] = a
        return a


class _Field:
    """One field of a dataset: an array plus its layout, with per-sample extraction to canonical ([Z,] Y, X) order."""

    def __init__(self, name: str, arr, layout: str, channel: Optional[int], squeeze_z: bool = True, transpose: bool = False):
        self.transpose = transpose
        self.name, self.arr, self.layout, self.channel, self.squeeze_z = name, arr, norm_layout(layout), channel, squeeze_z
        if len(self.layout) != arr.ndim:
            raise AdapterError(f"field '{name}': layout '{layout}' has {len(self.layout)} axes but the array has shape "
                               f"{tuple(arr.shape)}")
        if 'C' in self.layout and channel is None:
            raise AdapterError(f"field '{name}': layout has a channel axis C; give a 'channel' index")
        if 'C' in self.layout:
            nch = arr.shape[self.layout.index('C')]
            if not (-nch <= channel < nch):
                raise IndexError(f"channel {channel} out of range (array has {nch})")
        self.n = arr.shape[self.layout.index('N')] if 'N' in self.layout else 1
        rest = [c for c in self.layout if c not in 'NC']
        self.order = [c for c in 'ZYX' if c in rest]
        self._perm = [rest.index(c) for c in self.order]
        shp = [arr.shape[self.layout.index(c)] for c in self.order]
        self.shape = tuple(shp[1:]) if (squeeze_z and self.order[0] == 'Z' and shp[0] == 1) else tuple(shp)
        if transpose:
            self.shape = self.shape[:-2] + (self.shape[-1], self.shape[-2])

    def sample(self, i: int, offset: int = 0) -> np.ndarray:
        idx = []
        for c in self.layout:
            idx.append((i + offset) if c == 'N' else self.channel if c == 'C' else slice(None))
        a = np.asarray(self.arr[tuple(idx)])
        a = np.transpose(a, self._perm)
        if self.squeeze_z and self.order[0] == 'Z' and a.shape[0] == 1:
            a = a[0]
        return np.swapaxes(a, -1, -2) if self.transpose else a


def _spec(cfg: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    s = cfg.get(name)
    if s is None:
        return None
    if isinstance(s, str):
        s = {'key': s}
    s = dict(s)
    s.setdefault('layout', cfg.get('layout'))
    return s


def _split_labels(cfg: Dict[str, Any], n: int):
    sp = cfg.get('splits', 'none')
    if sp in (None, 'none', False):
        return lambda i, name='': None
    if sp == 'from_filename':
        def by_name(i, name=''):
            m = re.search(r'(?:^|[_\-.])(train|val|valid|validation|test)(?:[_\-.]|$)', name.lower())
            return None if not m else {'valid': 'val', 'validation': 'val'}.get(m.group(1), m.group(1))
        return by_name
    if isinstance(sp, dict) and sp.get('rule') == 'ictherm':       # IC-ThermBench: train_ratio 0.8, val_fraction 0.9, no shuffle
        trainval = max(2, min(int(n * 0.8), n - 1))
        n_train = max(1, min(int(trainval * 0.9), trainval - 1))
        bounds = {'train': (0, n_train), 'val': (n_train, trainval), 'test': (trainval, n)}
    elif isinstance(sp, dict):
        bounds = {k: (int(lo * n), int(hi * n)) for k, (lo, hi) in sp.items()}
    else:
        raise AdapterError(f'bad splits spec: {sp!r}')
    return lambda i, name='': next((k for k, (lo, hi) in bounds.items() if lo <= i < hi), None)


# ── dataset builders ─────────────────────────────────────────────────────────────
def open_dataset(root, cfg: Dict[str, Any]) -> Dataset:
    """Build a Dataset from `root` (file or directory) and a mapping config."""
    root = Path(root)
    if not root.exists():
        raise AdapterError(f'no such path: {root}')
    fmt = (cfg.get('format') or 'auto').lower()
    if fmt == '3dice':
        return _open_3dice(root, cfg)
    if 'u' not in cfg:
        raise AdapterError("config must map the output field: give 'u' (CLI: --u KEY)")
    specs = {nm: _spec(cfg, nm) for nm in FIELDS}
    per_file = root.is_dir() and not any(s and s.get('file') for s in specs.values())
    if per_file:
        return _open_per_file(root, cfg, specs)
    return _open_stacks(root, cfg, specs)


def _scalars(cfg):
    sp = cfg.get('spacing')
    return dict(units=cfg.get('units', ''), ambient=cfg.get('ambient'),
                spacing=None if sp is None else tuple(None if v is None else float(v) for v in sp))


def _open_stacks(root: Path, cfg, specs) -> Dataset:
    files: Dict[Path, _File] = {}
    fields: Dict[str, _Field] = {}
    for nm, s in specs.items():
        if s is None:
            continue
        path = root / s['file'] if s.get('file') else root
        if path.is_dir():
            raise AdapterError(f"field '{nm}' needs a 'file' (the dataset path {root} is a directory)")
        if not path.exists():
            if s.get('optional'):
                continue
            raise AdapterError(f"field '{nm}': no such file {path}")
        f = files.get(path) or files.setdefault(path, _File(path))
        key = s.get('key')
        if key is None and f.kind != 'npy':
            raise AdapterError(f"field '{nm}': give a 'key' (file keys: {f.keys()})")
        if s.get('layout') is None:
            raise AdapterError(f"field '{nm}': give a 'layout' (e.g. NHW)")
        try:
            fields[nm] = _Field(nm, f.get(key), s['layout'], s.get('channel'), transpose=bool(s.get('transpose')))
        except (AdapterError, IndexError) as exc:
            if s.get('optional') and (isinstance(exc, IndexError) or 'not in' in str(exc)):
                continue
            raise
    ns = {nm: f.n for nm, f in fields.items()}
    if len(set(ns.values())) > 1:
        raise AdapterError(f'fields disagree on the number of samples: {ns}')
    n = fields['u'].n
    sc = _scalars(cfg)
    label = _split_labels(cfg, n)
    name = cfg.get('name') or root.name

    def load(i):
        kw = {nm: fields[nm].sample(i) for nm in ('u', 'source', 'k', 'dirichlet_mask') if nm in fields}
        if 'dirichlet_mask' in kw:
            kw['dirichlet_mask'] = kw['dirichlet_mask'] > 0
        fo = fields['flux_out'].arr[i] if 'flux_out' in fields else None
        return Sample(**kw, spacing=sc['spacing'], ambient=sc['ambient'], id=f'{name}[{i}]', split=label(i),
                      flux_out=None if fo is None else float(np.asarray(fo).ravel()[0]))

    shapes = {nm: f.shape for nm, f in fields.items() if nm not in ('flux_out',)}
    ds = Dataset(n, load, name=name, units=sc['units'], shapes=lambda: shapes)
    ds._keep = files                                                # keep file handles alive
    return ds


def _open_per_file(root: Path, cfg, specs) -> Dataset:
    pattern = cfg.get('glob') or '*.npz'
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise AdapterError(f'no files matching {pattern} under {root}')
    sc = _scalars(cfg)
    n = len(paths)
    label = _split_labels(cfg, n)

    def load(i):
        f = _File(paths[i])
        kw = {}
        for nm in ('u', 'source', 'k', 'dirichlet_mask'):
            s = specs.get(nm)
            if s is None:
                continue
            try:
                kw[nm] = _Field(nm, f.get(s.get('key')), s['layout'], s.get('channel'), transpose=bool(s.get('transpose'))).sample(0)
            except (AdapterError, IndexError):
                if s.get('optional'):
                    continue
                raise
        if 'dirichlet_mask' in kw:
            kw['dirichlet_mask'] = kw['dirichlet_mask'] > 0
        fo = specs.get('flux_out')
        flux = float(np.asarray(f.get(fo['key'])).ravel()[0]) if fo else None
        return Sample(**kw, spacing=sc['spacing'], ambient=sc['ambient'], id=paths[i].stem, split=label(i, paths[i].stem), flux_out=flux)

    first = load(0)
    shapes = {'u': first.u.shape, **({'source': first.source.shape} if first.source is not None else {}),
              **({'k': first.k.shape} if first.k is not None else {})}
    return Dataset(n, load, name=cfg.get('name') or root.name, units=sc['units'], shapes=lambda: shapes)


def _open_3dice(root: Path, cfg) -> Dataset:
    """This repo's 3D-ICE .npz files: coords (n,3) [um], temp (n,) [K], power (n,), metadata (pickled dict).

    Scattered node coordinates are scattered onto the (z, y, x) tensor grid they form; a file whose nodes are not a full
    tensor grid fails to load and is reported by F001. Pickled metadata is read with allow_pickle (trusted local files only).
    """
    paths = sorted(root.rglob('*.npz')) if root.is_dir() else [root]
    if not paths:
        raise AdapterError(f'no .npz files under {root}')
    label = _split_labels({'splits': cfg.get('splits', 'from_filename')}, len(paths))

    def load(i):
        d = np.load(paths[i], allow_pickle=True)
        coords, temp, power = np.asarray(d['coords'], float), np.asarray(d['temp'], float), np.asarray(d['power'], float)
        if coords.ndim != 2 or coords.shape[1] != 3 or not (len(coords) == len(temp) == len(power)):
            raise ValueError(f'array lengths disagree: coords {coords.shape}, temp {temp.shape}, power {power.shape}')
        ux, uy, uz = (np.unique(coords[:, j]) for j in range(3))
        if len(ux) * len(uy) * len(uz) != len(coords):
            raise ValueError(f'{len(coords)} nodes are not a full {len(uz)} x {len(uy)} x {len(ux)} tensor grid')
        iz, iy, ix = (np.searchsorted(u, coords[:, j]) for u, j in ((uz, 2), (uy, 1), (ux, 0)))
        u = np.empty((len(uz), len(uy), len(ux)))
        s = np.empty_like(u)
        u[iz, iy, ix], s[iz, iy, ix] = temp, power
        meta = d['metadata'].item() if 'metadata' in d.files else {}
        dz = np.diff(uz)
        sp = (float(dz[0]) if len(dz) and np.allclose(dz, dz[0], rtol=1e-3) else None,
              float(np.diff(uy).mean()) if len(uy) > 1 else None, float(np.diff(ux).mean()) if len(ux) > 1 else None)
        amb = meta.get('t_ambient_kelvin')
        return Sample(u=u, source=s, spacing=sp, ambient=None if amb is None else float(amb), id=paths[i].stem,
                      split=label(i, paths[i].stem), meta={k: v for k, v in meta.items() if isinstance(v, (str, int, float))})

    return Dataset(len(paths), load, name=cfg.get('name') or root.name, units=cfg.get('units', 'K'))
