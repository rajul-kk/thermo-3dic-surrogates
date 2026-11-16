"""Projection (label-swap) test for the 3D-ICE die-edge artefact, report section 9.32.

The implementation lives in `tools/fieldlint` (`fieldlint.labelswap`) so that it can run on any paired artefact /
corrected label set; this module keeps the names the Kaggle kernels, scripts and tests import. A code snapshot for a
Kaggle kernel must therefore ship `tools/fieldlint/src` beside `src/` and `scripts/` (the October snapshots held only
the latter two).
"""
import sys
from pathlib import Path

try:
    import fieldlint.labelswap  # noqa: F401
except ImportError:  # no installed fieldlint, or an older one without labelswap: use the in-repo copy
    for name in [m for m in sys.modules if m == 'fieldlint' or m.startswith('fieldlint.')]:
        del sys.modules[name]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'fieldlint' / 'src'))

from fieldlint.labelswap import (  # noqa: F401
    FIELDS,
    RULE,
    SUPPORT_K,
    analyse,
    boot_ci,
    decide,
    per_scenario_arrays,
    records_from_summary,
    scenario_record,
    score,
    wilcoxon_p,
)
