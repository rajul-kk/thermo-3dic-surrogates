# Label-swap test: did a model learn an artefact in its training labels?

`fieldlint.labelswap` (CLI `fieldlint-labelswap`). It started as the paired projection test of the thermal benchmark's
report, section 9.32, where it showed that ThermoNO's apparent win over a physics solver came from reproducing a 3D-ICE
die-edge artefact. This page specifies it so that it runs on any paired "artefact / corrected" label set.

## Inputs

| input | shape | meaning |
|---|---|---|
| `T_old` | (n, ...) | held-out labels as shipped (artefact present) |
| `T_new` | (n, ...) | the same samples' corrected labels |
| `pred_old` | (seeds, n, ...) | held-out predictions of models trained on the old labels |
| `pred_new` | (seeds, n, ...) | the same, trained on the corrected labels |

Both arms must use **identical inputs**, architecture, hyper-parameters, folds and seeds. Only the training labels
differ. With cross-validation, each sample's prediction comes from the fold that held it out. At least 2 seeds per arm
are needed for the null, and 3 are recommended.

Stored results can be re-analysed without predictions: `--summary` reads the per-sample scores of a
`projection_summary.json` (`records_from_summary`).

## Statistic

For held-out sample i, the true artefact is `D = T_old - T_new` and a learned difference is `L = O(x) - N(x)`.

* `beta = <L, D> / <D, D>`, the projection slope. It is 1 when the artefact is reproduced exactly and 0 when L is
  orthogonal to D. A label-independent model error cancels in O - N, so beta stays meaningful for an inaccurate model.
* `corr`, the cosine between L and D.
* `beta_s` and `corr_s`, the same on the support `|D| > support_k`. The default 0.1 is in the label's units (K for the
  thermal data); set `--support-k` for other units.

The primary field is `beta`; the other three are secondary.

## Null and decision rule (unchanged from section 9.32)

The null is the same statistic between two same-label models, `N_a - N_b` (and `O_a - O_b`): what seed noise alone
gives. CIs are 95% bootstrap over held-out samples (4000 resamples, seed 0, so results are bit-reproducible).

* **learned**: the beta CI lies above the null CI and its lower bound is > 0.2.
* **not learned**: the CI upper bound is < 0.1 and the CI overlaps the null CI.
* **inconclusive**: otherwise.

The headline compares each seed singly, `O_k - N_k`, with a matched same-label pair: both are single-seed differences,
so their noise is matched. Every seed must agree. Seed k is paired with null pair `k mod n_pairs`. With 3 seeds that
pairs seed k with the k-th of the 3 pairs; with 2 seeds both seeds share the one pair. The seed-mean beta (equal to the
average of the per-seed betas, by linearity) is reported beside it. Its own decision against the single-seed null is
conservative against "learned", because averaging lowers its noise below the null's.

## Power

A single seed calls "learned" only when its lower CI bound clears both the null's upper bound and 0.2. If the
per-sample spread of beta under an effect matches the null's (sd), the smallest true beta it detects with about 50%
power is

    MDB = max(0.2, null mean + z sd / sqrt(n)) + z sd / sqrt(n),    z = 1.96

(`min_detectable_beta`). Add a further 0.84 sd / sqrt(n) for about 80% power. Compute it from a pilot's same-label pairs
before registering n.

**Worked example (section 9.32's stored nulls, n = 45 each).**

| unit | null sd | MDB per null pair | observed seed-mean beta | decision |
|---|---|---|---|---|
| ThermoNO geometry4 | 0.30 | 0.31, 0.31, 0.25 | 0.95 | learned |
| ThermoNO geometry5 | 0.06 | 0.22, 0.22, 0.21 | 0.77 | learned |
| ThermoNO geometry6 | 0.03 | 0.21, 0.21, 0.21 | 0.80 | learned |
| FNO geometry4 | 1.04 | 0.65, 0.60, 0.54 | 0.35 | inconclusive |
| FNO geometry5 | 2.16 | 1.66, 1.67, 1.46 | 0.06 | inconclusive |
| FNO geometry6 | 1.30 | 0.85, 0.92, 0.65 | 0.26 | inconclusive |
| CNO-FNO-attn geometry5 | 1.86 | 1.12 | 0.06 | inconclusive |
| CNO-FNO-attn geometry6 | 1.11 | 0.84 | 0.34 | inconclusive |

The FNO-family rows are inconclusive for lack of power, not because nothing was learned. Their seed noise is 3 to 70
times ThermoNO's, so with 45 samples no beta under about 0.5 could have been called. Detecting FNO geometry4's 0.35 at
its seed noise would need about (1.96 x 1.04 / 0.075)^2, roughly 740 held-out samples. So a design should either have
low seed noise or enough samples. Report the MDB with every verdict.

## Reproduction of section 9.32

`tests/test_label_swap_reproduction.py` (repository root) recomputes the analysis of every unit from the stored
per-sample scores in `results/artefact_projection/projection_summary.json`. It matches the stored analysis exactly,
including the betas and decisions of the section 9.32 table. The stored file holds per-sample scores, not prediction
fields, so the reproduced stage is the inference (`analyse`, `decide`) and not `score`. `score` and `scenario_record`
are covered by the synthetic tests in `tools/fieldlint/tests/test_labelswap.py`.

## Limits

* It needs a corrected label set. A leaked split or a documented design property has none.
* It is uninformative when D is as large as the field, as with a pure relabelling such as a transpose: any model that
  fits its labels gives beta near 1. It is built for artefacts that are small against the field.
* beta measures reproduction along D only. A model can learn a different spurious structure that this test does not
  see.
* The decision thresholds 0.2 and 0.1 were fixed for the thermal study. Changing them for an application must be
  registered beforehand.

## Related work (checked 2026-11-16)

* Patel, arXiv 2610.09255 (7 Oct 2026, NeurIPS 2026 AI for Science workshop). Trains identical networks on two
  numerical schemes and reports their difference as a share of the full-imitation ceiling. Uses a Fourier-symbol probe
  of the linearised one-step operator; linear advection, plus Burgers at the operator level. That is the closest
  design. Differences here: the second arm is a corrected label set rather than another scheme, the null is seed noise,
  the decision rule is pre-registered, and it works on predictions of any model, without linearising it.
* Koehler and Thuerey, arXiv 2510.23111: emulators can beat their low-fidelity training solver in rollouts. A different
  question, accuracy against a reference, not artefact attribution.
* Kim, Mitchell and Wang, arXiv 2602.10457: relabel a few training examples and retrain, to probe fairness. The same
  retrain-on-changed-labels idea, outside PDEs.
