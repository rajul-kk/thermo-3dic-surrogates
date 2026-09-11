"""Molecular featurisers. The featuriser is treated as part of the model, since 'which fingerprint' is an axis the literature disagrees on."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List

import numpy as np

log = logging.getLogger(__name__)


def _mols(smiles: List[str]):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    return [Chem.MolFromSmiles(s) for s in smiles]


def morgan(smiles: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP-style circular fingerprints -- the baseline representation most papers use."""
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, m in enumerate(_mols(smiles)):
        if m is not None:
            out[i] = gen.GetFingerprintAsNumPy(m).astype(np.float32)
    return out


def maccs(smiles: List[str]) -> np.ndarray:
    """167-bit MACCS structural keys."""
    from rdkit.Chem import MACCSkeys
    out = np.zeros((len(smiles), 167), dtype=np.float32)
    for i, m in enumerate(_mols(smiles)):
        if m is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(m)
        for b in fp.GetOnBits():
            out[i, b] = 1.0
    return out


def descriptors(smiles: List[str]) -> np.ndarray:
    """RDKit physicochemical descriptors, NaN/inf-cleaned."""
    from rdkit.Chem import Descriptors
    names = [n for n, _ in Descriptors.descList]
    calc = {n: f for n, f in Descriptors.descList}
    out = np.zeros((len(smiles), len(names)), dtype=np.float64)
    for i, m in enumerate(_mols(smiles)):
        if m is None:
            continue
        for j, n in enumerate(names):
            try:
                v = calc[n](m)
            except Exception:
                v = 0.0
            out[i, j] = v
    # Clip BEFORE the cast: RDKit's Ipc is routinely ~1e60, which overflows to +inf in
    # float32. Cleaning in float64 and then casting re-introduces the inf the clean removed,
    # and sklearn then rejects every trial that sampled a descriptor featuriser.
    out[~np.isfinite(out)] = 0.0
    lim = np.finfo(np.float32).max / 4.0
    np.clip(out, -lim, lim, out=out)
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def morgan_plus_descriptors(smiles: List[str]) -> np.ndarray:
    """ECFP concatenated with physicochemical descriptors."""
    return np.hstack([morgan(smiles), descriptors(smiles)])


FEATURISERS: Dict[str, Callable[[List[str]], np.ndarray]] = {
    'morgan': morgan,
    'maccs': maccs,
    'descriptors': descriptors,
    'morgan+desc': morgan_plus_descriptors,
}
