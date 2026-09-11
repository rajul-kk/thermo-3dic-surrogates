"""Published numbers, quoted not reproduced. Every entry carries its source and split protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# All values transcribed 2026-09-11 from Table 1 of FP-GNN (arXiv:2205.03834), which
# aggregates its own results with those it attributes to Chemprop [Yang et al.] and to
# Wu et al. for Attentive FP / HRGCN+ / XGBoost. Read directly from the PDF table, not from
# a secondary summary. Blank cells in that table are recorded as None rather than guessed.


@dataclass(frozen=True)
class Published:
    dataset: str
    split: str            # 'scaffold' | 'random' -- as *labelled* by the source
    metric: str           # 'roc_auc' | 'rmse'
    model: str
    value: float
    kind: str             # 'neural' | 'non-neural'
    source: str
    note: Optional[str] = None


P: List[Published] = [
    # --- BBBP, scaffold. Note the 22.6-point spread under one split label. ---
    Published('bbbp', 'scaffold', 'roc_auc', 'MoleculeNet (GraphConv)', 0.690, 'neural',
              'arXiv:2205.03834 Table 1', 'attributed to the MoleculeNet benchmark'),
    Published('bbbp', 'scaffold', 'roc_auc', 'Chemprop (optimized)', 0.886, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bbbp', 'scaffold', 'roc_auc', 'FP-GNN', 0.916, 'neural',
              'arXiv:2205.03834 Table 1'),
    # --- BBBP, random ---
    Published('bbbp', 'random', 'roc_auc', 'Chemprop (optimized)', 0.917, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bbbp', 'random', 'roc_auc', 'Attentive FP', 0.887, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bbbp', 'random', 'roc_auc', 'HRGCN+', 0.926, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bbbp', 'random', 'roc_auc', 'XGBoost', 0.926, 'non-neural',
              'arXiv:2205.03834 Table 1', 'best of Wu et al.; a non-neural model ties the best GNN'),
    Published('bbbp', 'random', 'roc_auc', 'FP-GNN', 0.935, 'neural',
              'arXiv:2205.03834 Table 1'),
    # --- BACE, scaffold ---
    Published('bace', 'scaffold', 'roc_auc', 'MoleculeNet (Weave)', 0.806, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bace', 'scaffold', 'roc_auc', 'Chemprop (optimized)', 0.857, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bace', 'scaffold', 'roc_auc', 'FP-GNN', 0.860, 'neural',
              'arXiv:2205.03834 Table 1'),
    # --- BACE, random ---
    Published('bace', 'random', 'roc_auc', 'Chemprop (optimized)', 0.898, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bace', 'random', 'roc_auc', 'Attentive FP', 0.876, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bace', 'random', 'roc_auc', 'HRGCN+', 0.891, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('bace', 'random', 'roc_auc', 'XGBoost', 0.889, 'non-neural',
              'arXiv:2205.03834 Table 1', 'best of Wu et al.'),
    Published('bace', 'random', 'roc_auc', 'FP-GNN', 0.881, 'neural',
              'arXiv:2205.03834 Table 1', 'FP-GNN does NOT win here'),
    # --- FreeSolv, random (RMSE, lower better) ---
    Published('freesolv', 'random', 'rmse', 'MoleculeNet (MPNN)', 1.150, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('freesolv', 'random', 'rmse', 'Chemprop (optimized)', 1.009, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('freesolv', 'random', 'rmse', 'Attentive FP', 1.091, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('freesolv', 'random', 'rmse', 'HRGCN+', 0.926, 'neural',
              'arXiv:2205.03834 Table 1', 'best published value recorded here'),
    Published('freesolv', 'random', 'rmse', 'XGBoost', 1.025, 'non-neural',
              'arXiv:2205.03834 Table 1'),
    Published('freesolv', 'random', 'rmse', 'FP-GNN', 0.905, 'neural',
              'arXiv:2205.03834 Table 1'),
    # --- ESOL, random (RMSE, lower better). The proposing paper's own model is LAST here. ---
    Published('esol', 'random', 'rmse', 'MoleculeNet (MPNN)', 0.580, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('esol', 'random', 'rmse', 'Chemprop (optimized)', 0.587, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('esol', 'random', 'rmse', 'Attentive FP', 0.587, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('esol', 'random', 'rmse', 'HRGCN+', 0.563, 'neural',
              'arXiv:2205.03834 Table 1', 'best published value'),
    Published('esol', 'random', 'rmse', 'XGBoost', 0.582, 'non-neural',
              'arXiv:2205.03834 Table 1', 'beats Chemprop, Attentive FP and FP-GNN'),
    Published('esol', 'random', 'rmse', 'FP-GNN', 0.675, 'neural',
              'arXiv:2205.03834 Table 1', 'worst of the six in the proposing paper own table'),
    # --- Lipophilicity, random (RMSE, lower better) ---
    Published('lipo', 'random', 'rmse', 'MoleculeNet (GraphConv)', 0.655, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('lipo', 'random', 'rmse', 'Chemprop (optimized)', 0.563, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('lipo', 'random', 'rmse', 'Attentive FP', 0.553, 'neural',
              'arXiv:2205.03834 Table 1', 'best published value'),
    Published('lipo', 'random', 'rmse', 'HRGCN+', 0.603, 'neural',
              'arXiv:2205.03834 Table 1'),
    Published('lipo', 'random', 'rmse', 'XGBoost', 0.574, 'non-neural',
              'arXiv:2205.03834 Table 1', 'beats HRGCN+ and FP-GNN'),
    Published('lipo', 'random', 'rmse', 'FP-GNN', 0.625, 'neural',
              'arXiv:2205.03834 Table 1'),
]

# Where a non-neural model already wins in the published tables themselves.
NONNEURAL_WINS_IN_PUBLISHED = """Read the quoted table rather than its abstract and the picture is already mixed, before this
audit runs a single model. In FP-GNN's own Table 1 (arXiv:2205.03834), XGBoost -- a
non-neural baseline -- beats the paper's proposed architecture on:

  ESOL (random, RMSE)          XGBoost 0.582  vs  FP-GNN 0.675
  Lipophilicity (random, RMSE) XGBoost 0.574  vs  FP-GNN 0.625
  BBBP (random, ROC-AUC)       XGBoost 0.926  ties HRGCN+ 0.926, beats Chemprop 0.917

and on BACE (random) FP-GNN (0.881) is beaten by XGBoost (0.889), HRGCN+ (0.891) and
Chemprop (0.898). The contradiction this audit exists to adjudicate is therefore visible
inside single papers, not only between them."""

# The single most important thing in this file.
SPLIT_LABEL_WARNING = """\
Published values labelled 'scaffold' are NOT mutually comparable. On BBBP the same split
label spans 0.690 (MoleculeNet/DeepChem) to 0.916 (FP-GNN) -- a 22.6-point range. This
project measured a 19-point swing (RF: 0.705 -> 0.895) on BBBP from changing only the
tie-break rule between equal-sized Bemis-Murcko scaffold groups, with model, features and
data held fixed (molprop/README.md 2b). That is very nearly the whole published spread.

Consequence: comparing our numbers to a published number requires matching the tie-break
convention, not just the word 'scaffold'. Use split='scaffold_det' against MoleculeNet-derived
values. We have NOT confirmed which convention Chemprop or FP-GNN used, so no head-to-head
against those two is asserted here."""


def for_cell(dataset: str, split: str, metric: str) -> List[Published]:
    """Published rows matching one audit cell, best-first."""
    rows = [p for p in P if p.dataset == dataset and p.split == split and p.metric == metric]
    return sorted(rows, key=lambda p: p.value, reverse=(metric == 'roc_auc'))


def best(dataset: str, split: str, metric: str) -> Optional[Published]:
    rows = for_cell(dataset, split, metric)
    return rows[0] if rows else None


def summary(dataset: str, split: str, metric: str) -> str:
    """One-line reference for the audit's console output."""
    # 'scaffold_det' is our deterministic split; published 'scaffold' rows are the closest
    # available comparison, but see SPLIT_LABEL_WARNING before treating it as like-for-like.
    lookup = 'scaffold' if split == 'scaffold_det' else split
    rows = for_cell(dataset, lookup, metric)
    if not rows:
        return 'published reference: none recorded for this cell'
    lo, hi = min(r.value for r in rows), max(r.value for r in rows)
    b = rows[0]
    tag = ' [split convention unverified]' if lookup == 'scaffold' else ''
    return (f'published ({lookup}): best {b.model} {b.value:.3f}, '
            f'range {lo:.3f}-{hi:.3f} over {len(rows)} models{tag}')
