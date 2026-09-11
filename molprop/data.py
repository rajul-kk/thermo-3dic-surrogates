"""MoleculeNet loading, caching, and the two split types the literature disagrees over."""
from __future__ import annotations

import gzip
import io
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
CACHE = Path(__file__).resolve().parent / 'cache'

# name -> (remote file, smiles column, label column(s), task type)
DATASETS: Dict[str, Tuple[str, str, List[str], str]] = {
    'bbbp':    ('BBBP.csv', 'smiles', ['p_np'], 'classification'),
    'bace':    ('bace.csv', 'mol', ['Class'], 'classification'),
    'clintox': ('clintox.csv.gz', 'smiles', ['FDA_APPROVED', 'CT_TOX'], 'classification'),
    'esol':    ('delaney-processed.csv', 'smiles',
                ['measured log solubility in mols per litre'], 'regression'),
    'freesolv': ('SAMPL.csv', 'smiles', ['expt'], 'regression'),
    'lipo':    ('Lipophilicity.csv', 'smiles', ['exp'], 'regression'),
}


@dataclass
class Dataset:
    """One MoleculeNet task: parsed SMILES, labels, and the task type."""
    name: str
    smiles: List[str]
    y: np.ndarray          # (n,) or (n, n_tasks)
    task: str              # 'classification' | 'regression'

    @property
    def n_tasks(self) -> int:
        return 1 if self.y.ndim == 1 else self.y.shape[1]


def _download(fname: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / fname
    if not dest.exists():
        log.info('downloading %s', fname)
        urllib.request.urlretrieve(BASE + fname, dest)
    return dest


def load(name: str) -> Dataset:
    """Download (once), parse, and drop molecules RDKit cannot read."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    fname, smi_col, label_cols, task = DATASETS[name]
    path = _download(fname)
    if fname.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    keep_smiles, keep_rows = [], []
    for i, smi in enumerate(df[smi_col].astype(str).tolist()):
        if Chem.MolFromSmiles(smi) is not None:
            keep_smiles.append(smi)
            keep_rows.append(i)
    dropped = len(df) - len(keep_rows)
    if dropped:
        log.info('%s: dropped %d/%d unparseable SMILES', name, dropped, len(df))

    y = df.iloc[keep_rows][label_cols].to_numpy(dtype=float)
    if y.shape[1] == 1:
        y = y.ravel()
    return Dataset(name, keep_smiles, y, task)


def scaffold_split(smiles: List[str], frac: Tuple[float, float, float],
                   seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis-Murcko scaffold split: whole scaffold groups go to one side, never split."""
    # The MoleculeNet standard and the harder of the two splits. Groups are assigned
    # largest-first to train, which is the usual deterministic convention; `seed` only
    # shuffles same-size groups so repeated runs are not identical.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smi), includeChirality=False)
        except Exception:
            scaf = smi
        groups.setdefault(scaf, []).append(i)

    rng = np.random.default_rng(seed)
    sets = list(groups.values())
    order = sorted(range(len(sets)),
                   key=lambda k: (-len(sets[k]), rng.random()))
    n = len(smiles)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    tr, va, te = [], [], []
    for k in order:
        g = sets[k]
        if len(tr) + len(g) <= n_tr:
            tr += g
        elif len(va) + len(g) <= n_va:
            va += g
        else:
            te += g
    return np.array(tr), np.array(va), np.array(te)


def scaffold_split_deterministic(smiles: List[str], frac: Tuple[float, float, float]
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DeepChem's ScaffoldSplitter ordering: no tie-breaking, so the split is fixed.

    This is the split published MoleculeNet numbers are computed on, and it is far harder
    than the randomised-tie-break version above: same model and features, BBBP RF scores
    0.705 here against 0.895 there (see molprop/README.md). It takes no seed, because it
    has no randomness -- which is itself the point.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    groups: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smi), includeChirality=False)
        except Exception:
            scaf = smi
        groups.setdefault(scaf, []).append(i)

    # Sort by group size descending, ties by first appearance -- the deterministic convention.
    # Train is filled first, so test receives the smallest groups. The tie term must be
    # `g[0]`, not `-g[0]`: under reverse=True the negated form inverts the tie order and, on
    # BBBP, hands the test set every positive molecule (pos_rate 1.000), which makes ROC-AUC
    # undefined and silently NaNs the whole cell.
    sets = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    tr, va, te = [], [], []
    for g in sets:
        if len(tr) + len(g) > n_tr:
            if len(va) + len(g) > n_va:
                te += g
            else:
                va += g
        else:
            tr += g
    return np.array(tr), np.array(va), np.array(te)


def random_split(n: int, frac: Tuple[float, float, float],
                 seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform random split -- the easier setting, and a prime suspect for the disagreement."""
    idx = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def make_split(ds: Dataset, kind: str, seed: int,
               frac: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """Dispatch to scaffold or random splitting."""
    if kind == 'scaffold':
        return scaffold_split(ds.smiles, frac, seed)
    if kind == 'scaffold_det':
        return scaffold_split_deterministic(ds.smiles, frac)
    if kind == 'random':
        return random_split(len(ds.smiles), frac, seed)
    raise ValueError(f'unknown split {kind!r}')
