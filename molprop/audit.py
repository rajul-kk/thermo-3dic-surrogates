"""CLI for the molecular property prediction baseline audit. See README.md."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from molprop import data as D
from molprop import published
from molprop.protocol import Result, run_cell, separable

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('molprop')

DEFAULT_MODELS = ['trivial', 'linear', 'knn', 'rf', 'xgboost', 'lightgbm']
DEFAULT_FEATS = ['morgan', 'maccs', 'descriptors', 'morgan+desc']


def main():
    ap = argparse.ArgumentParser(description='Baseline audit for molecular property prediction')
    ap.add_argument('--datasets', nargs='+', default=['bbbp', 'bace', 'esol'],
                    choices=sorted(D.DATASETS))
    ap.add_argument('--splits', nargs='+', default=['scaffold', 'random'],
                    choices=['scaffold', 'scaffold_det', 'random'])
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    ap.add_argument('--featurisers', nargs='+', default=DEFAULT_FEATS)
    ap.add_argument('--budget', type=int, default=24,
                    help='hyperparameter trials per model -- IDENTICAL for every model, which '
                         'is the control that makes the comparison about method not effort')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--output', type=Path, default=Path('molprop/results/audit.json'))
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    results = []
    for name in args.datasets:
        ds = D.load(name)
        log.info('%s: %d molecules, task=%s, %d label column(s)',
                 name, len(ds.smiles), ds.task, ds.n_tasks)
        for split in args.splits:
            cells = {}
            for m in args.models:
                cells[m] = run_cell(ds, split, m, seeds=seeds, budget=args.budget,
                                    featurisers=args.featurisers)
                results.append(cells[m])

            metric = 'AUC (higher better)' if ds.task == 'classification' else 'RMSE (lower better)'
            print(f'\n=== {name} / {split} split / {metric} ===')
            print(f'{"model":<10}{"mean":>9}{"std":>8}{"featuriser":>14}   vs trivial')
            triv = cells.get('trivial')
            ordered = sorted(cells.values(),
                             key=lambda r: (-r.mean if ds.task == 'classification' else r.mean))
            for r in ordered:
                verdict = ''
                if triv is not None and r.model != 'trivial':
                    verdict = ('separable' if separable(r, triv)
                               else 'NOT separable from trivial')
                print(f'{r.model:<10}{r.mean:>9.4f}{r.std:>8.4f}{str(r.featuriser):>14}   {verdict}')
            best = ordered[0]
            others = [r for r in ordered[1:] if r.model != 'trivial']
            ties = [r.model for r in others if not separable(best, r)]
            print(f'best: {best.model}' + (f'  (not separable from: {", ".join(ties)})' if ties else ''))
            metric = 'roc_auc' if ds.task == 'classification' else 'rmse'
            rows = published.for_cell(name, 'scaffold' if split == 'scaffold_det' else split,
                                      metric)
            if rows:
                print('published for reference (quoted, not reproduced):')
                for r in rows:
                    kind = 'non-neural' if r.kind == 'non-neural' else ''
                    print(f'  {r.model:<24}{r.value:>8.3f}  {kind:<11}{r.source}')
                if split in ('scaffold', 'scaffold_det'):
                    # Published 'scaffold' numbers differ by up to 22.6 points on BBBP purely
                    # by tie-break convention, so this is not a like-for-like comparison.
                    print('  [!] published "scaffold" rows are NOT mutually comparable; '
                          'see published.SPLIT_LABEL_WARNING')
            else:
                print('published reference: none recorded for this cell '
                      '(see molprop/published.py)')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        # failed_trials/budget_unmatched are persisted, not just logged: a cell whose budget
        # could not be filled is not comparable, and that must survive into the artifact
        # rather than living only in a console line nobody re-reads.
        json.dump([{'dataset': r.dataset, 'split': r.split, 'model': r.model,
                    'task': r.task, 'scores': r.scores, 'mean': r.mean, 'std': r.std,
                    'featuriser': r.featuriser, 'seconds': r.seconds,
                    'failed_trials': r.failed_trials,
                    'budget_unmatched': r.budget_unmatched} for r in results],
                  f, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
