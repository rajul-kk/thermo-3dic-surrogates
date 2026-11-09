"""fieldlint: physics-aware linter for PDE-surrogate benchmark datasets (steady diffusion / heat-type fields)."""
__version__ = '0.1.0'

from .adapters import AdapterError, load_config, open_dataset  # noqa: E402
from .model import Dataset, Sample, dataset_from_arrays  # noqa: E402
from .rules import RULES, Finding, Options, RuleResult  # noqa: E402
from .runner import Report, run  # noqa: E402

__all__ = ['AdapterError', 'Dataset', 'Sample', 'Finding', 'Options', 'RULES', 'Report', 'RuleResult',
           'dataset_from_arrays', 'load_config', 'open_dataset', 'run']
