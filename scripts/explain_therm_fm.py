"""
Therm-FM weight-drift XAI: diff a pretrained CNOFNOHybrid checkpoint against
its few-shot fine-tuned result to see WHAT changed to adapt to a new
geometry.

Pure post-hoc checkpoint analysis — no forward passes, no retraining, no
compute cost beyond loading two .pt files. Bucketed by the same parameter
groups finetune_therm_fm.py freezes/unfreezes (encoder, film_gen, decoder,
proj, tail latent_blocks), so drift in the frozen encoder should be exactly
zero — this doubles as a correctness check that freezing actually worked.

Usage
-----
python scripts/explain_therm_fm.py \
    --pretrained checkpoints/cno_fno/cno_fno_best.pt \
    --finetuned  checkpoints/therm_fm/therm_fm_best.pt \
    --output     results/therm_fm_drift
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_therm_fm import _ENCODER_PREFIXES, _DECODER_PREFIXES, _TAIL_BLOCK_ATTR

_log = logging.getLogger(__name__)


def _group_for(name: str) -> str:
    """Bucket a parameter name into the same groups finetune_therm_fm.py uses."""
    if name.startswith(_ENCODER_PREFIXES):
        return 'encoder (frozen)'
    if name.startswith('film_gen'):
        return 'film_gen'
    if name.startswith(_DECODER_PREFIXES):
        return 'decoder'
    if name.startswith('proj'):
        return 'proj'
    if name.startswith(_TAIL_BLOCK_ATTR):
        return 'latent_blocks (tail + frozen-middle mixed)'
    return 'other'


_NEAR_ZERO_INIT_FLOOR = 1e-4  # base_norm below this is treated as "zero-initialized"


def compute_weight_drift(
    pretrained_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
) -> List[Dict]:
    """
    Per-parameter drift: L2 norm of the difference (absolute) and relative
    drift (||delta|| / ||pretrained||). Returns a list of per-parameter
    records, each also tagged with its coarse group.

    Some parameters (e.g. FiLM's final layer, GroupNorm biases) are
    deliberately zero-initialized so the layer starts as an identity
    transform. For these, ||pretrained|| ~ 0 makes relative drift explode
    to a meaningless huge number even for a tiny absolute update. Such
    tensors are flagged 'near_zero_init' and excluded from relative-drift
    aggregation — absolute drift is the only meaningful metric for them.
    """
    records = []
    common_keys = sorted(set(pretrained_sd) & set(finetuned_sd))
    missing_ft = sorted(set(pretrained_sd) - set(finetuned_sd))
    missing_pt = sorted(set(finetuned_sd) - set(pretrained_sd))
    if missing_ft:
        _log.warning("%d keys in pretrained but not finetuned (skipped): %s",
                     len(missing_ft), missing_ft[:5])
    if missing_pt:
        _log.warning("%d keys in finetuned but not pretrained (skipped): %s",
                     len(missing_pt), missing_pt[:5])

    for key in common_keys:
        w0 = pretrained_sd[key]
        w1 = finetuned_sd[key]
        # Some spectral-conv weights (e.g. SpectralConv3d in the latent FNO
        # blocks) are complex-valued (torch.cfloat). .float() on a complex
        # tensor silently discards the imaginary part rather than computing
        # magnitude, which corrupts the norm. .norm() natively supports
        # complex tensors (returns a real-valued modulus-based norm), so we
        # only cast real tensors and leave complex ones as-is.
        if not w0.is_complex():
            w0 = w0.float()
        if not w1.is_complex():
            w1 = w1.float()
        if w0.shape != w1.shape:
            _log.warning("Shape mismatch at %s: %s vs %s -- skipped", key, w0.shape, w1.shape)
            continue

        delta_norm = (w1 - w0).norm().item()
        base_norm = w0.norm().item()
        near_zero_init = base_norm < _NEAR_ZERO_INIT_FLOOR
        rel_drift = float('nan') if near_zero_init else delta_norm / base_norm

        records.append({
            'name': key,
            'group': _group_for(key),
            'n_params': w0.numel(),
            'delta_norm': delta_norm,
            'base_norm': base_norm,
            'rel_drift': rel_drift,
            'near_zero_init': near_zero_init,
        })
    return records


def summarize_by_group(records: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-parameter drift records into per-group summary stats."""
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        groups.setdefault(r['group'], []).append(r)

    summary = {}
    for group, recs in groups.items():
        total_params = sum(r['n_params'] for r in recs)
        total_delta_norm = sum(r['delta_norm'] for r in recs)
        total_base_norm = sum(r['base_norm'] for r in recs)

        finite_recs = [r for r in recs if not r['near_zero_init']]
        n_excluded = len(recs) - len(finite_recs)
        finite_params = sum(r['n_params'] for r in finite_recs)
        if finite_params > 0:
            weighted_rel_drift = sum(r['rel_drift'] * r['n_params'] for r in finite_recs) / finite_params
            max_rel_drift = max(r['rel_drift'] for r in finite_recs)
        else:
            weighted_rel_drift = float('nan')
            max_rel_drift = float('nan')

        summary[group] = {
            'n_tensors': len(recs),
            'n_params': total_params,
            'n_excluded_near_zero_init': n_excluded,
            'mean_rel_drift': weighted_rel_drift,
            'max_rel_drift': max_rel_drift,
            'total_abs_drift_norm': total_delta_norm,
            'total_base_norm': total_base_norm,
        }
    return summary


def plot_drift_summary(summary: Dict[str, Dict], output_path: Path) -> None:
    """Two-panel bar chart: absolute drift norm (always defined) and mean
    relative drift (NaN for groups made entirely of near-zero-init tensors,
    shown as a gap rather than a misleading zero or huge spike)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import math

    groups = sorted(summary.keys(), key=lambda g: -summary[g]['total_abs_drift_norm'])
    abs_drift = [summary[g]['total_abs_drift_norm'] for g in groups]
    rel_drift = [summary[g]['mean_rel_drift'] for g in groups]
    rel_drift_plot = [0.0 if math.isnan(v) else v for v in rel_drift]
    rel_is_nan = [math.isnan(v) for v in rel_drift]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(range(len(groups)), abs_drift, color='tab:blue')
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups, rotation=25, ha='right')
    axes[0].set_ylabel('sum ||W_finetuned - W_pretrained||  (absolute)')
    axes[0].set_title('Absolute weight drift by group')
    axes[0].grid(axis='y', alpha=0.3)

    bars = axes[1].bar(range(len(groups)), rel_drift_plot, color='tab:orange')
    for i, is_nan in enumerate(rel_is_nan):
        if is_nan:
            axes[1].text(i, 0.01, 'n/a\n(zero-init)', ha='center', va='bottom',
                        fontsize=8, rotation=90, color='gray')
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups, rotation=25, ha='right')
    axes[1].set_ylabel('mean ||delta|| / ||pretrained|| (param-weighted)')
    axes[1].set_title('Relative weight drift by group')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('Therm-FM weight drift: what changed to adapt to the new geometry')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    _log.info("Saved drift plot: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Therm-FM weight-drift XAI")
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--finetuned', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

    args.output.mkdir(parents=True, exist_ok=True)

    pt_ckpt = torch.load(args.pretrained, map_location='cpu')
    ft_ckpt = torch.load(args.finetuned, map_location='cpu')
    # Accept either checkpoint key convention (see note in finetune_therm_fm.py):
    # FNOTrainer saves use 'model_state', this repo's other scripts use 'model'.
    pt_sd = pt_ckpt.get('model') or pt_ckpt.get('model_state') or pt_ckpt
    ft_sd = ft_ckpt.get('model') or ft_ckpt.get('model_state') or ft_ckpt

    records = compute_weight_drift(pt_sd, ft_sd)
    summary = summarize_by_group(records)

    _log.info("Weight drift summary by group:")
    import math
    for group, stats in sorted(summary.items(), key=lambda kv: kv[1]['total_abs_drift_norm'], reverse=True):
        rel_str = 'n/a (all near-zero-init)' if math.isnan(stats['mean_rel_drift']) else f"{stats['mean_rel_drift']:.6f}"
        excl_note = f" ({stats['n_excluded_near_zero_init']} near-zero-init tensors excluded from rel. drift)" \
            if stats['n_excluded_near_zero_init'] else ""
        _log.info(
            "  %-45s n_tensors=%-3d n_params=%-9d abs_drift=%.4f  mean_rel_drift=%s%s",
            group, stats['n_tensors'], stats['n_params'],
            stats['total_abs_drift_norm'], rel_str, excl_note,
        )

    encoder_group = summary.get('encoder (frozen)')
    if encoder_group is not None:
        if encoder_group['mean_rel_drift'] > 1e-8:
            _log.warning(
                "Encoder group shows nonzero drift (%.2e) despite being frozen -- "
                "check that freeze_encoder() was actually called before fine-tuning.",
                encoder_group['mean_rel_drift'],
            )
        else:
            _log.info("Encoder drift is exactly zero, as expected -- freeze worked correctly.")

    plot_drift_summary(summary, args.output / 'weight_drift_by_group.png')

    import json
    with open(args.output / 'weight_drift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    _log.info("Saved summary: %s", args.output / 'weight_drift_summary.json')


if __name__ == '__main__':
    main()
