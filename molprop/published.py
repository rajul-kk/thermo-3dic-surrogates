"""Published numbers to compare against, quoted not reproduced -- the same stance as report Sec 9.13."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# (dataset, split) -> {label: (score, source)}
#
# These are transcribed from the papers, NOT reproduced here. They are the comparison
# target: this audit supplies the missing well-tuned non-neural row, it does not re-run
# GNNs. Anything whose split or metric could not be confirmed is left out rather than
# guessed -- a mis-transcribed reference number would invalidate the whole comparison,
# which is exactly the failure that produced the stale-baseline correction in
# docs/report.md Sec 9.12.
PUBLISHED: Dict[Tuple[str, str], Dict[str, Tuple[float, str]]] = {
    # Intentionally empty until each number is checked against its paper's own text,
    # including which split and metric it used. Populate via `add()` below with a citation.
}


def add(dataset: str, split: str, label: str, score: float, source: str) -> None:
    """Record a published number with its citation. Verify the split/metric first."""
    PUBLISHED.setdefault((dataset, split), {})[label] = (score, source)


def get(dataset: str, split: str) -> Dict[str, Tuple[float, str]]:
    return PUBLISHED.get((dataset, split), {})
