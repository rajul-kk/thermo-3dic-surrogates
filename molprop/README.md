# molprop — a baseline audit for molecular property prediction

**Status:** runs complete, 2026-09-12. **Read §8 first: a prior-art check found both headline claims were published in 2024.** Results in §7 below; the deterministic-scaffold pass is still running at time of writing.

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

**The contradiction is visible inside single papers, not only between them.** In FP-GNN's own
Table 1 (arXiv:2205.03834), XGBoost — a non-neural baseline — beats the paper's proposed
architecture on ESOL (0.582 vs 0.675 RMSE) and Lipophilicity (0.574 vs 0.625 RMSE), and ties
the best model on BBBP random (0.926). On BACE random, FP-GNN (0.881) is beaten by XGBoost,
HRGCN+ and Chemprop alike. These are quoted from the proposing paper, not measured here; see
`published.py`.

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

| dataset | task | n (verified) | metric | in the current run? |
|---|---|---|---|---|
| BBBP | binary classification | 2,039 | ROC-AUC | **yes** |
| BACE | binary classification | 1,513 | ROC-AUC | **yes** |
| ESOL (delaney) | regression | 1,128 | RMSE | **yes** |
| FreeSolv (SAMPL) | regression | 642 | RMSE | **yes** |
| ClinTox | binary classification (2 tasks) | 1,480 | ROC-AUC + PRC-AUC | **yes** (§7.6) |
| Lipophilicity (`lipo`) | regression | 4,200 | RMSE | **yes** (§7.6) |

The first four are what the reported run covers; ClinTox and Lipophilicity are implemented and
loadable but were left out to keep the matched-budget sweep inside a CPU-only time budget
(6 models × 4 featurisers × 24 trials × 5 seeds × 2 splits, ~3 h as it stands). They are named
here so the scope is explicit rather than implied — Lipophilicity in particular is the largest
of the six and would roughly double the cost.

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

## 2b. A protocol result found while building this: "scaffold split" is not one split

Measured 2026-09-11, before any model comparison was run.

Bemis-Murcko scaffold splitting groups molecules by scaffold and assigns whole groups to
train/val/test. Most scaffolds are singletons (BBBP: 1,025 scaffold groups for 2,039
molecules), so **the order in which equal-sized groups are assigned decides the split**, and
that choice is not usually stated in papers.

Two defensible conventions, same data, same scaffold-disjointness, same model (RF, 400 trees,
Morgan r=2/2048), BBBP:

| scaffold-split convention | BBBP test ROC-AUC |
|---|---|
| deterministic, ties by file order (DeepChem `ScaffoldSplitter`) | **0.705** |
| randomised tie-break, 5 seeds | **0.895 ± 0.013** |

**A 19-point AUC gap from the tie-break rule alone** — larger than most architecture gaps the
literature reports, and roughly 14× the seed spread, so no amount of seed averaging within one
convention reveals it.

> **Narrowed by the full run (§7.6).** This 19-point figure is **BBBP-specific** and was
> stated too broadly here at first. Across all four datasets the convention shifts results by
> 0.003–0.21, **in a direction that varies by dataset**: much harder on BBBP, roughly neutral
> on BACE, and *easier* on ESOL. The defensible claim is not "deterministic is harder" but
> that the convention moves the number by an amount comparable to architecture differences,
> with a sign a reader cannot predict from the split's name.

> **Scooped — see §8.1.** MOLTOP (arXiv:2407.12136) App. G reported this in 2024, on the same
> dataset and at the same magnitude ("as much as 20% on BBBP"). What follows is an independent
> reproduction, not a discovery, and must not be cited as one.

**This is not merely our problem: it reproduces a spread that already exists in the published
literature.** FP-GNN's Table 1 (arXiv:2205.03834) lists three BBBP results all labelled
*scaffold*:

| source (all labelled "scaffold split") | BBBP ROC-AUC |
|---|---|
| MoleculeNet (GraphConv), DeepChem splitter | **0.690** |
| Chemprop (optimized) | 0.886 |
| FP-GNN | **0.916** |

a **22.6-point published spread under one split label** — and our two tie-break conventions
(0.705 / 0.895), with model, features and data held fixed, very nearly bracket it. The
deterministic value also lands within 0.02 of the MoleculeNet row, which is what one would
expect if MoleculeNet's DeepChem-derived split is the deterministic convention.

We do **not** claim this explains the whole published spread — we have not confirmed which
convention Chemprop and FP-GNN used, and `published.py` records that explicitly rather than
asserting a head-to-head. The defensible claim is narrower and still strong: *a difference of
the same magnitude as the entire published spread is obtainable without changing the model at
all.*

Consequences for this audit, adopted as rules:

1. **`scaffold_det` is the split to use when comparing against published numbers.** The
   randomised variant (`scaffold`) is statistically better — it has a variance estimate at all
   — but its absolute numbers are **not comparable to published tables**, and this README's
   earlier framing that they were is wrong.
2. Both are reported, because the gap between them is itself a result: it bounds how much of
   a published "scaffold split" difference can be protocol rather than method.
3. This strengthens the audit's premise rather than undermining it. The literature
   disagreement in §1 is exactly the shape of thing an unstated tie-break rule produces.

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

## 7. Results — first full run (2026-09-11)

4 datasets x 2 splits x 6 models x 4 featurisers, budget 24 trials, 5 seeds, CPU only.
**Zero cells hit `BUDGET NOT MATCHED`**, so the matched-budget control held throughout.
Raw: `molprop/results/audit_scaffold_random.json` (gitignored).

"Separable" below means the gap exceeds the **sum** of the two standard deviations — the
deliberately conservative test of §3.3.

| dataset / split | best model | is the best separable from `linear`? |
|---|---|---|
| bbbp / scaffold | lightgbm 0.9259 | **yes** — linear is separably worse |
| bbbp / random | rf 0.9299 | **no** |
| bace / scaffold | xgboost 0.8770 | **no** |
| bace / random | **linear 0.9064** | **no** — the linear model *is* the best |
| esol / scaffold | lightgbm 0.8234 | **yes** |
| esol / random | xgboost 0.5356 | **yes** |
| freesolv / scaffold | xgboost 2.0097 | no — but see the instability note below |
| freesolv / random | lightgbm 0.9441 | **no** |

### 7.1 The headline

**On three of the four classification cells, a tuned logistic regression on fingerprints is
statistically indistinguishable from tuned gradient-boosted trees** — and on BACE/random it is
the single best model. The exception is BBBP/scaffold, where LightGBM (0.9259) beats linear
(0.8804) by 0.0455 against a combined spread of 0.0394, so linear is separably worse there.
(An earlier draft of this section said all four; that was a misreading of the tie-list, which
excludes `linear` on that cell.)

This is the outcome §4 called most likely and most useful. Published differences of ~0.02 AUC
on BBBP or BACE are not evidence about architecture: our entire six-model spread on BBBP/random
is 0.012 AUC, *narrower than the 0.887–0.935 spread across the five published architectures*
for the same dataset and split type.

### 7.2 "Not separable" means two different things, and they should not be conflated

- **BBBP, BACE, FreeSolv/random** — the models genuinely perform alike. Small gaps, small
  spreads.
- **FreeSolv/scaffold** — `linear` trails xgboost by 0.75 RMSE, a large gap, but its seed
  spread is **±1.22**, larger than the gap itself. It is not that the two are equivalent; it
  is that a 642-molecule scaffold split cannot rank the linear model at all. A single-seed
  paper could show linear winning or losing by ~2 RMSE here purely by draw.

Reporting only "not separable" would hide that distinction, so both are stated.

### 7.3 Where the non-neural baselines beat published neural results

Comparable only where the split type matches and the split has no tie-break ambiguity, i.e.
the **random** cells. Published values from `published.py` (FP-GNN Table 1, arXiv:2205.03834).

| cell | our best non-neural | best published | published worst |
|---|---|---|---|
| esol / random (RMSE) | **xgboost 0.5356** | HRGCN+ 0.563 | FP-GNN 0.675 |
| freesolv / random (RMSE) | lightgbm 0.9441 | FP-GNN 0.905 | MoleculeNet/MPNN 1.150 |
| bbbp / random (AUC) | rf 0.9299 | FP-GNN 0.935 | Attentive FP 0.887 |
| bace / random (AUC) | **linear 0.9064** | Chemprop 0.898 | FP-GNN 0.881 |

On **ESOL/random our tuned XGBoost beats every published model**, neural included. The
sharpest single data point: Wu et al. report XGBoost at 0.582 on this exact cell and we get
0.536 with the same model family — **a 0.046 improvement from tuning alone, which exceeds the
gap between four of the six published models.** That is the audit's thesis in its own terms:
a baseline's published number is substantially a function of how hard someone tuned it.

On **BACE/random our logistic regression (0.9064) exceeds all four published values.**

Caveat stated plainly: random splits differ by seed between studies, and our ±0.02–0.04 is the
same order as the published spread. The defensible claim is that **our non-neural baselines
land at or above the top of the published range**, not that any specific model was beaten by a
specific margin.

### 7.4 What the featuriser selection shows

On both ESOL cells and FreeSolv/random every tree model independently selected a
descriptor-containing featuriser (`morgan+desc` or `descriptors`), while `linear` and `knn`
fell back to `maccs`. Physicochemical descriptors carry real signal for solubility and
hydration free energy, and only the tree models exploit it. That is a mechanism for the ESOL
gap, not merely a number — and it is visible only because the featuriser was searched under
the same budget as the hyperparameters (§3.2) rather than fixed in advance.

### 7.6 Extension: ClinTox, Lipophilicity, and the metric axis (2026-09-12)

The first run covered 2 classification and 2 regression datasets. Claims of the form
"classification benchmarks do not separate models" rested on **n=2 datasets**, which is thin,
and §1's *metric* suspect had never been tested at all. Both are now addressed.

**ClinTox** (1,480 molecules, two tasks with opposite imbalance: task 0 FDA_APPROVED 93.6%
positive, task 1 CT_TOX 7.6% positive) and **Lipophilicity** (4,200, the largest set here).
Zero `BUDGET NOT MATCHED` warnings across all runs.

| cell | best | linear | linear separably worse? |
|---|---|---|---|
| clintox-t0 / scaffold | lightgbm 0.9181 | 0.8391 | **yes** |
| clintox-t0 / random | rf 0.9196 | 0.8737 | no |
| clintox-t1 / scaffold | lightgbm 0.9163 | 0.8827 | no |
| clintox-t1 / random | rf 0.9244 | 0.8606 | no |
| lipo / scaffold (RMSE) | xgboost 0.7347 | 0.9260 | **yes** |
| lipo / random (RMSE) | lightgbm 0.6262 | 0.8023 | **yes** |

**(1) The classification claim is now sharper, and split type is the axis that matters.**
Across all eight seeded classification cells, linear is **never separably worse on a random
split** (0 of 4 — and on BACE/random it is the *best* model), but is separably worse on 2 of 4
scaffold splits. The original framing ("classification does not separate") was too coarse: the
harder structural-novelty split is where tuned trees earn their advantage, and the easier
random split is where logistic regression on fingerprints is indistinguishable from gradient
boosting. That is a more useful statement and it needed the third dataset to see.

**(2) Regression separates on 4 of 6 cells, and FreeSolv is the lone exception.** ESOL (both
splits) and Lipophilicity (both splits) separate cleanly; FreeSolv does not, and it is by far
the smallest set (642) with a linear-model seed spread of ±1.22. So "regression separates" is
supported, with dataset size the plausible moderator rather than task type alone.

**(3) The metric axis changes the ranking — tested directly.** ClinTox task 1 (7.6% positive),
identical data, splits, seeds and budget; only the metric differs:

| split | ROC-AUC order | PRC-AUC order |
|---|---|---|
| scaffold | **lightgbm** > rf > xgboost > linear | **rf** > xgboost > lightgbm > linear |
| random | rf > **xgboost** > lightgbm > linear | rf > lightgbm > **linear** > xgboost |

On the random split **XGBoost falls from 2nd to 4th, below the linear model**, purely from the
metric change. Two further effects: PRC-AUC has an **informative floor** (the trivial baseline
scores the positive rate, 0.069–0.092, where ROC-AUC scores exactly 0.5000 ± 0.0000 by
construction and tells you nothing), and PRC-AUC is **far noisier** here (relative spreads
24–33% against ~4%). Under both metrics linear remains not separable from the best, so that
conclusion is metric-robust; what is not robust is the ordering among the non-linear models.
A paper reporting one metric on one seed could name three different winners on this cell.

**(4) Where our baselines do NOT reach the published range — stated because it bounds the
claim.** On Lipophilicity/random our best tuned baseline is **0.6262**, against published
Attentive FP 0.553, Chemprop 0.563, **XGBoost 0.574**, HRGCN+ 0.603 and FP-GNN 0.625. We sit
at the *bottom* of that range and are beaten by the published XGBoost row — the opposite of
ESOL/random, where our XGBoost (0.536) beat every published model. The likely cause is that a
24-trial budget is not enough for the largest dataset here, which is a limitation of this
audit rather than evidence about the models. It is reported because an audit that only
published its favourable comparisons would be committing the error it exists to expose.

### 7.5 What this does not show

- ClinTox and Lipophilicity were not run (§2).
- No GNN was trained here. Neural numbers are quoted from their papers, as §3.5 states.
  A GNN tuned as hard as these baselines might well separate on the classification sets.
- The scaffold cells above use the **randomised** tie-break and are therefore *not*
  comparable to published scaffold numbers. See §2b — that is what the `scaffold_det` pass
  exists to supply.

### 7.6 The same benchmarks under the deterministic scaffold convention

Second pass, `--splits scaffold_det`, same budget/seeds/featurisers. Raw:
`molprop/results/audit_scaffold_det.json`. 4/4 blocks clean, no budget failures.

**`scaffold_det` is seedless.** Every seed sees the same split, so its std captures
model-seed variance *only* — not split variance. Separability verdicts in this section are
therefore **much weaker evidence** than those in §7, and the two must not be compared. The
audit prints a `[!]` caution on every such block.

| dataset | model | randomised scaffold | deterministic scaffold | Δ |
|---|---|---|---|---|
| bbbp (AUC ↑) | rf | 0.9122 | **0.7554** | −0.157 |
| | xgboost | 0.9189 | 0.7319 | −0.187 |
| | lightgbm | 0.9259 | 0.7189 | −0.207 |
| | linear | 0.8804 | 0.7011 | −0.179 |
| | knn | 0.8987 | 0.6905 | −0.208 |
| bace (AUC ↑) | rf | 0.8698 | 0.8729 | **+0.003** |
| | xgboost | 0.8770 | 0.8626 | −0.014 |
| | lightgbm | 0.8576 | 0.8593 | +0.002 |
| | knn | 0.8371 | 0.8469 | +0.010 |
| | linear | 0.8647 | 0.8250 | −0.040 |
| esol (RMSE ↓) | lightgbm | 0.8234 | **0.7577** | −0.066 (better) |
| | rf | 0.8554 | 0.7642 | −0.091 (better) |
| | xgboost | 0.8570 | 0.8077 | −0.049 (better) |
| | linear | 1.2420 | 0.9292 | −0.313 (better) |
| | knn | 1.4318 | 1.5386 | +0.107 (worse) |
| freesolv (RMSE ↓) | lightgbm | 2.0450 | 2.0717 | +0.027 |
| | xgboost | 2.0097 | 2.1962 | +0.187 |
| | rf | 2.3703 | 2.1954 | −0.175 |
| | knn | 2.9130 | 2.1862 | −0.727 (better) |
| | linear | 2.7614 | 3.0290 | +0.268 |

**The convention is not uniformly harder — it is unpredictable.** BBBP loses 0.16–0.21 AUC
across every model; BACE barely moves (≤0.04); ESOL gets *easier* for four of five models;
FreeSolv moves in both directions depending on model. A paper that writes "scaffold split"
without naming the convention has not pinned down the difficulty of its own benchmark, and a
reader cannot infer even the sign of the bias.

**The mechanism, measured.** The deterministic convention induces a label-distribution shift
that the randomised one does not:

| dataset | overall pos-rate | deterministic test | randomised test |
|---|---|---|---|
| BBBP | 0.765 | **0.522** (shift −0.243) | 0.767 (+0.002) |
| BACE | 0.457 | 0.605 (shift +0.149) | 0.387 (−0.070) |

BBBP's deterministic "hard scaffold split" is hard substantially because its test set is
near-balanced while the data is 77% positive — a label shift, not purely a structural-novelty
test. This predicts what we observe: BACE, already near-balanced, shifts and moves less. It
also means BBBP scaffold results are measuring something other than what the split is usually
described as measuring.

### 7.7 Like-for-like against published numbers

`scaffold_det` is the convention published MoleculeNet-derived numbers appear to use (§2b), so
these are the closest available comparisons. Both caveats above still apply.

| cell | our best non-neural | published neural |
|---|---|---|
| bace / scaffold (AUC ↑) | **rf 0.8729**, xgboost 0.8626, lightgbm 0.8593 | FP-GNN 0.860, Chemprop 0.857, MoleculeNet/Weave 0.806 |
| bbbp / scaffold (AUC ↑) | rf 0.7554, xgboost 0.7319 | FP-GNN 0.916, Chemprop 0.886, MoleculeNet/GraphConv 0.690 |

**On BACE our tuned random forest beats all three published architectures.** On BBBP it does
not — our best baseline (0.755) sits above MoleculeNet's GraphConv (0.690) but well below
Chemprop (0.886) and FP-GNN (0.916). We do **not** read that as "GNNs win on BBBP": those two
higher numbers are consistent with the randomised convention (our randomised baselines reach
0.88–0.93, squarely in their range), and §2b shows the convention alone spans that gap. The
honest conclusion is that **the BBBP comparison is unresolvable from published information**,
because the convention is not reported. That is the finding, not a defeat.

### 7.8 Summary of what this audit establishes

1. On **three of four** classification cells with a seeded split, a tuned logistic regression
   is **not separable** from tuned gradient-boosted trees; on BACE/random it is the best
   model. The exception is BBBP/scaffold, where it is separably worse.
2. On **regression** (ESOL both splits) trees **are** separable from linear and kNN, with a
   mechanism — only trees exploit the physicochemical descriptors they all selected.
   So the non-separability above is a property of those benchmarks, not a blunt test.
3. Tuning moves a baseline by more than published architecture gaps: our XGBoost on ESOL
   random (0.536) beats Wu et al.'s XGBoost (0.582) by more than the spread separating four
   of the six published models — and beats every published model on that cell.
4. The unreported **scaffold tie-break convention** shifts results by up to 0.21 AUC in a
   dataset-dependent direction, which is enough to make published "scaffold split"
   comparisons unresolvable where the convention is not stated.

None of this requires training a GNN, which is the point: these are the controls the
comparison needed, and they cost CPU-hours.

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

---

## 8. Prior-art check: both headline claims were already published (2026-09-12)

Run deliberately, after the results were in and before any of this was presented as novel.
It went badly for this audit, and the outcome is recorded here rather than quietly dropped.

### 8.1 The scaffold tie-break finding is fully scooped

§2b reports that the Bemis-Murcko tie-break convention moves BBBP by ~18 AUC points and that
published "scaffold split" numbers are therefore not mutually comparable.

**MOLTOP** (arXiv:2407.12136, ECAI 2024), Appendix G, already reports this: the deterministic
vs balanced distinction "makes a very significant difference in scores... often by 5% or as
much as 20% **on BBBP dataset**" — the same dataset and the same effect size we measured. It
further identifies GROVER and D-MPNN as having used non-deterministic splits, making their
numbers incomparable to deterministic protocols.

The distinction is also documented in **scikit-fingerprints**, which quantifies it on BACE
(78.25% deterministic vs 85.33% randomised) and states that "performance estimation is much
more optimistic in this variant".

So §2b is a **reproduction**, not a discovery. It is kept because independent reproduction of a
protocol artefact has some value, and because the mechanism below may add something — but it
must not be cited as new.

### 8.2 The "non-neural baselines are competitive" finding is scooped

**MOLTOP** is a Random Forest over topological descriptors, evaluated on 8 MoleculeNet
datasets using OGB's deterministic scaffold splits with 10 seeds and reported standard
deviations, where "only GEM outperforms MOLTOP significantly" (Wilcoxon signed-rank,
p = 0.016). That is the same claim §7 makes, under a comparable protocol, published first.

### 8.3 What may still be unreported

Stated conservatively, and none of it is a headline:

1. **The mechanism behind BBBP's sensitivity.** MOLTOP quantifies the effect; we measured
   *why* — the deterministic convention gives BBBP a test set at 52.2% positive against 76.5%
   overall, a 24-point label shift (§7.6). We have not found that explanation published.
2. **A trivial-baseline row.** MOLTOP reports no majority-class baseline; its weakest entry is
   a fingerprint model at 62.6% AUROC. We report `trivial` on every cell, which is what makes
   the PRC-AUC floor visible (0.069–0.092 versus ROC-AUC's structural 0.5000).
3. **Matched tuning budget across model families.** MOLTOP is deliberately
   hyperparameter-free, so it does not address the "the proposed model was tuned and the
   baseline was not" confound. §3.2 does.
4. **The PRC-AUC ranking flip** on ClinTox task 1 (XGBoost 2nd → 4th, below linear). Metric
   sensitivity on imbalanced data is textbook; this specific demonstration on a MoleculeNet
   cell under matched budget we have not seen.

### 8.4 What this means for the project

This directory was built because §1 found the published literature self-contradictory about
whether baselines or GNNs win. A prior-art check should have come first: MOLTOP had already
adjudicated much of it in 2024. The lesson is the same one the thermal work keeps producing —
**check what exists before measuring** — and it applies to us here rather than to someone else.

The work retains value as an independent reproduction with a trivial baseline, a matched
budget, and a metric-axis test, and `docs/report.md` §9.16f states this in the paper itself so
no reader mistakes it for a novel result.
