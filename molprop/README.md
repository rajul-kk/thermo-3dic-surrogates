# molprop — a baseline audit for molecular property prediction

**Status:** design + implementation, 2026-09-11. No results yet.

Self-contained: this directory does not import from the 3D-IC thermal code and does not
depend on it. It reuses the *method* that produced `docs/report.md` §9.13, not the code.

---

## 1. The question, and why it is open

There is an active, unresolved contradiction in the literature:

- Random forests and SVMs on ECFP fingerprints "consistently outperform recently developed
  methods" for small-molecule potency; gradient-boosted trees (CatBoost) with multiple
  fingerprints are stronger still.
- Other work reports that graph attention networks beat Random Forest on MoleculeNet
  "by a significant margin", and that fingerprint-enhanced GNNs beat all baselines on eight
  MoleculeNet datasets.

Both cannot be generally true. Contradictions of this shape almost always come from
differences in **protocol** rather than differences in **method**:

| suspect | why it would flip a result |
|---|---|
| split type | scaffold splits are much harder than random splits, and papers differ on which they use. A model tuned for one can look far better on the other. |
| tuning budget | if the proposed model gets a tuned search and the baseline gets library defaults, the comparison measures effort, not architecture. |
| metric | ROC-AUC, PRC-AUC and accuracy rank models differently on the heavily imbalanced MoleculeNet classification sets. |
| seed variance | several MoleculeNet sets have only 1-4k molecules; single-split differences of 0.02 AUC are frequently within noise. |
| trivial floor | almost nobody reports what a majority-class or train-mean predictor scores, so it is unclear how much of a reported AUC is real signal. |

**This audit does not propose a new model.** It fixes the protocol, runs the non-neural
baselines properly under it, and reports what survives. It is the same move as adding the
missing non-neural row to IC-ThermBench (`docs/report.md` §9.13), which is worth doing here
precisely because the field disagrees with itself.

## 2. What is measured

Six MoleculeNet tasks, chosen to be small (total ~4 MB), standard, and to span both task
types:

| dataset | task | n (approx) | metric |
|---|---|---|---|
| BBBP | binary classification | 2,039 | ROC-AUC |
| BACE | binary classification | 1,513 | ROC-AUC |
| ClinTox | binary classification (2 tasks) | 1,478 | ROC-AUC |
| ESOL (delaney) | regression | 1,128 | RMSE |
| FreeSolv (SAMPL) | regression | 642 | RMSE |
| Lipophilicity | regression | 4,200 | RMSE |

Models, all non-neural:

- **trivial** — majority class / train mean. The floor. Reported because it is almost never
  reported, and it bounds how much of a headline number is real.
- **ridge / logistic** — linear on fingerprints.
- **kNN** — Tanimoto-ish nearest neighbours on fingerprints.
- **random forest** — the classical strong baseline.
- **XGBoost**, **LightGBM** — gradient-boosted trees, the models that most often beat GNNs
  in the papers that find baselines winning.

Featurisations: Morgan/ECFP (radius 2, 2048 bits), MACCS keys (167), RDKit physicochemical
descriptors, and ECFP+descriptors concatenated. The featuriser is treated as part of the
model and selected under the same budget, since "which fingerprint" is exactly the axis the
fingerprint-vs-GNN papers differ on.

## 3. The protocol, which is the actual contribution

Every control below exists because its absence is a plausible explanation for the published
disagreement.

1. **Both split types, same seeds.** Scaffold (Bemis-Murcko, the MoleculeNet standard) and
   random. Reporting both makes split choice visible instead of load-bearing.
2. **Matched tuning budget.** Every model gets an identical number of random-search trials
   on an identical validation split. No model gets defaults while another gets a search.
3. **Repeated seeds with dispersion.** Five seeds per configuration, mean ± std reported. A
   difference smaller than the seed spread is reported as "not separable", not as a win.
4. **Never tuned on test.** Hyperparameters are chosen on validation; test is touched once.
5. **Published numbers are quoted, not reproduced.** GNN results come from their papers and
   are labelled as such. We are supplying the missing baseline row, not re-running GNNs —
   the same stance as §9.13, and it is what makes this cheap.

## 4. What a result looks like

Three outcomes, all publishable, which is the point of choosing this question:

- **Baselines win under matched protocol** → the GNN results in the literature are partly
  a tuning/split artefact. Strongest outcome.
- **GNNs win under matched protocol** → the fingerprint-advocacy papers were under-tuned,
  and the audit says so. Also a real result.
- **Not separable within seed variance** → the most likely outcome on the small datasets,
  and arguably the most useful: it means published differences of ~0.02 AUC on BBBP/BACE
  are not evidence of anything.

## 5. Compute

CPU only, minutes per dataset. No GPU. The largest cost is fingerprint generation, which is
seconds. This is deliberate: the whole argument is that the missing baseline is cheap.

## 6. Layout

```
molprop/
  README.md      this file
  data.py        MoleculeNet download, caching, scaffold/random splits
  features.py    Morgan/ECFP, MACCS, RDKit descriptors
  models.py      the baseline model zoo + hyperparameter spaces
  protocol.py    matched-budget search, repeated seeds, metrics
  published.py   published GNN/baseline numbers, with citations
  audit.py       CLI entry point
```

Run:

```bash
python -m molprop.audit --datasets bbbp bace esol --splits scaffold random
```
