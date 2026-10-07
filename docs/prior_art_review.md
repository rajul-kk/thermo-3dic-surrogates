# Prior-art review of the main findings (2026-10-07)

Replaces the quick web check of 2026-10-06 (`notes/2026-10-06-gaps-plan.md`, "D results"). One search campaign per
finding. Verdicts are deliberately conservative.

**Provenance labels.** *found* = located by web search and the abstract or page was fetched this session.
*snippet* = seen only in a search-result snippet, not fetched. *inferred* = our own reasoning, no source.

**Limits of this review.** The search tool is a general web search, not a GitHub, PyPI or Semantic Scholar query
engine. "Nothing found" means nothing surfaced in roughly 35 queries and 25 page fetches, not that nothing exists.
Most fetches returned abstracts or landing pages, not full text; where a claim depends on full text it is marked.
Several 2026 arXiv items are recent and unreviewed.

## Method: queries used

1. Surrogates and solver artefacts: "neural surrogate inherits numerical error of training simulator solver bias";
   "neural operator ... learns discretization error artifacts of the solver; label noise structure in PDE surrogate
   datasets"; "ML emulator inherits biases of training simulation data ... mesh imprint"; "surrogate reproduces
   simulator artifact paired ablation retrain on corrected labels swap labels"; "Neural Emulator Superiority
   2510.23111"; "Mohan 2024 neural models learn numerical artifacts structural errors"; "ML turbulence closure ...
   learns numerical error of discretization scheme".
2. 3D-ICE `.lyt` issue: "3D-ICE github issue layout file .lyt partial cell material power overlap"; direct fetch of
   github.com/esl-epfl/3d-ice/issues (all issues, issue #3); fetch of the 3D-ICE 4.0 PDF (text extracted locally,
   searched for GDS, overlap, rho, tiles, `.lyt`).
3. Linear baselines: "linear baseline ridge regression outperforms neural operators ... PDEBench"; "thermal
   surrogate chip power map linear superposition baseline layout randomization DeepOHeat Therm-FM"; "linear
   regression POD baseline matches neural operator accuracy"; "DeepOHeat Therm-FM fixed floorplan ... limitation";
   fetches of IC-ThermBench (arXiv and GitHub), HSL benchmark 2103.11177, DAIST 2610.02461, PCA-RaNN 2606.29440.
4. Differential testing: "differential testing of PDE solvers finite element independent discretization";
   "differential testing thermal simulator ... HotSpot 3D-ICE discrepancy"; "metamorphic testing scientific software
   survey"; "code-to-code comparison differential testing verification of simulation codes"; "metamorphic relations
   heat conduction solver testing"; "N-version / cross-simulator ... neural surrogate training data verification".
5. Dataset linters: "data validation tool for scientific machine learning datasets PDE surrogate dataset quality
   checks linter leakage"; "neural operator benchmark dataset flaw ... trivial baseline"; "automated checks for
   simulation-generated datasets physics consistency maximum principle ... audit tool"; fetches of FlowMLLab,
   the PDE-simulator diagnostic suite, the SciML V&V paper, the metamorphic surrogate-testing paper.
6. IC-ThermBench transpose: "IC-ThermBench transposed power channel errata github issue"; "orientation axis
   transposition bug in scientific machine learning benchmark dataset"; fetches of arXiv abs page (versions),
   github.com/Day333/ThermalBench and its issue list.

## Finding 1. Surrogates reproduce a deterministic simulator artefact (paired label-swap projection test)

| Work | Overlap | Provenance |
|---|---|---|
| Mohan, Chattopadhyay & Miller (2024), "What You See is Not What You Get: Neural PDEs and the Illusion of Learning", [arXiv:2411.15101](https://arxiv.org/abs/2411.15101) | Abstract states neural-PDE models "learn the artifacts in the simulation training data arising from the discretized Taylor Series truncation error". The general claim is already made. Methods: numerical analysis, Jacobian analysis, weight eigenanalysis; no paired label-swap test in the abstract. Full text not read. | found |
| Koehler & Thuerey (2025), "Neural Emulator Superiority", NeurIPS 2025, [arXiv:2510.23111](https://arxiv.org/abs/2510.23111) | Converse: emulators can filter structured solver error in rollouts. Its related work itself cites the "emulators learn artefacts" failure mode (to Mohan et al.). All experiments are autoregressive rollouts, no steady-state setting, no paired artefact-projection test (checked in the HTML version). | found |
| Suriyanarayanan, Chakraborty & Maulik (2026), "Deep Learning of Solver-Aware Turbulence Closures from Nudged LES Dynamics", [arXiv:2604.23874](https://arxiv.org/abs/2604.23874) | Abstract: learned correction "systematically tracks the discretization error of the coarse solver". Different setting (LES closure); a related observation that learned models carry the discretization error. | found |
| Koehler (2026), "From Numerical Simulators of PDEs to Neural Emulators and Back", [arXiv:2608.24547](https://arxiv.org/abs/2608.24547) | Spectral analysis of numerical errors in emulators, unconverged-solver effects on training. PDF text could not be read; I cannot say whether it contains a reproduction test. | found (abstract only) |
| Northcutt et al. (2021), "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks", [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) | Label errors in general ML benchmarks. Not PDE-aware, not about deterministic solver structure. | found (snippet) |

**Verdict: PARTIALLY ANTICIPATED.** The general statement "surrogates trained on discretised simulation data can learn the
discretisation's artefacts" is published (Mohan et al. 2024), and Neural Emulator Superiority treats it as a known
failure mode. What remains ours: (a) a specific, mechanistically understood bug class (partial-cell power or material
inconsistency), not generic truncation error; (b) a steady-state, chip-thermal, non-rollout setting; (c) the paired
label-swap projection test with pre-registered thresholds, which gave a per-architecture answer (ThermoNO 77-95%,
FNO partial and below the bar). No prior paired test found, but I only read abstracts of the closest works.
The correct framing is "a clean causal measurement of a known phenomenon", not "a new phenomenon".

## Finding 2. 3D-ICE 4.0 `.lyt` route: material by cell centre, power by overlap area

| Work | Overlap | Provenance |
|---|---|---|
| Zhu, Huang, Costero & Atienza (2025), 3D-ICE 4.0, [arXiv:2512.05823](https://arxiv.org/abs/2512.05823) | Its GDS-import pipeline (Algorithm 1, quadtree tiles) computes per-tile intersection area with layout polygons and an overlap ratio rho used to build equivalent per-tile models. Text read locally. It does not discuss `.lyt` files or the cell-centre rule. | found |
| 3D-ICE GitHub issues, [esl-epfl/3d-ice](https://github.com/esl-epfl/3d-ice/issues) | Two issues only: #1 `.mlt` file usage (2021); #3 "Bugs that may exist" (2026-05), abnormal temperature with top and bottom heat sinks plus non-uniform mode. Neither concerns layout partial cells. | found |
| 3D-ICE user guide, [eslweb.epfl.ch](https://eslweb.epfl.ch/3Dice/3D-ICE-User_Guide.pdf) | Documents the layout format; not read for the partial-cell rule this session. | found (not read) |
| Cut-cell / volume-fraction treatments in FV/FD solvers (general numerics) | A partially filled cell should treat source and material consistently by volume fraction. Standard knowledge, no specific citation. | inferred |

**Verdict: NOVEL as a bug report; the framing needs softening.** No public report found, and the tracker is nearly
empty, so absence there is weak evidence. The earlier note called it "an internal inconsistency in 3D-ICE
(GDS route area-weighted, `.lyt` route not)". That is an inference: the 4.0 paper describes an external tile-extraction
pipeline, not the solver's `.lyt` reader, so the two are not shown to be alternative routes into the same solver.
Safer wording: "the GDS extraction in the 4.0 paper uses area-overlap equivalent materials, whereas the `.lyt`
reader (our reproduction, `warpage/scripts/edge_artifact_3dice.sh`) assigns material by cell centre". The reproduction
is ours; the maintainers' intent is unknown. It has not been reported upstream, which would be the cheap way to settle it.

**Update (2026-10-07): independent confirmation and a second, related observation.**
- *HotSpot confirms the artefact.* On the misaligned die-edge repro the peak rise is 14.562 K for 3D-ICE, 11.067 K for
  our independent finite-volume reference and 11.060 K for HotSpot (independent discretisation, `-detailed_3D` per-tile
  resistivity; report §9.34). Two independent solvers agree with each other and not with the `.lyt` route. Both use an
  area-fraction rule, so this shows that 3D-ICE's cell-centre assignment departs from area-fraction treatments, not
  what the maintainers intend.
- *Non-uniform mode ignores the `.lyt`.* In 3D-ICE 4.0 non-uniform mode (`non-uniform true ;` plus `discretization`)
  the layout file has no effect: a case with an SRC layer of GAP plus an SI rectangle in the `.lyt` gives a rise of
  12.849 K, identical to an all-GAP control (max |diff| = 0.0), against 7.507 K for the conforming fixture. We
  reproduced this directly. No warning is printed.
- *Prior report: not found, as an inference.* We did not find the `.lyt`-ignored behaviour in the 3D-ICE 4.0 paper
  (arXiv:2512.05823) or in the documentation we could read, and the issue tracker's non-uniform report (#3, abnormal
  temperatures with top and bottom heat sinks) is a different symptom. Absence from the sources we searched is not proof
  that it is unreported or undocumented; the user guide was not read for this rule. Verdict for this observation:
  NOVEL as a report, weak evidence, same standing as the main finding. A bottom-heat-sink segfault in non-uniform mode
  was seen once and is unverified; it is not claimed.

## Finding 3. Fixed-placement benchmarks linearly solvable; layout randomisation breaks it

| Work | Overlap | Provenance |
|---|---|---|
| McGreivy & Hakim (2024), *Nat. Mach. Intell.*, [arXiv:2407.07218](https://arxiv.org/abs/2407.07218) | 79% of ML-vs-numerical papers use weak baselines. General claim anticipated (already in references.md). Fluid PDEs only. | found (earlier session) |
| Hang et al. (2026), IC-ThermBench, [arXiv:2608.23977](https://arxiv.org/abs/2608.23977); [code](https://github.com/Day333/ThermalBench) | Layout, material and boundary-condition scopes S2-S5; eight neural baselines; no non-neural row. | found |
| Chen et al. (2021), heat-source-layout benchmark, [arXiv:2103.11177](https://arxiv.org/abs/2103.11177) | Ten DNNs on layout-to-temperature; abstract does not mention a regression baseline. | found (abstract) |
| Deng et al. (2026), PCA-RaNN, [arXiv:2606.29440](https://arxiv.org/abs/2606.29440) | PCA plus closed-form least-squares readout is "competitive" with operator-learning baselines on Burgers, Darcy, NS. A linear-readout-is-enough result for generic PDEs, not a benchmark critique. | found (abstract) |
| Classical superposition and Green's-function thermal methods (references.md, 2007 entry) | Chip thermal is linear in power for a fixed stack, so a linear fit solving fixed-placement data is expected. | earlier session |

**Verdict: PARTIALLY ANTICIPATED.** The mechanism is classical (linearity in power; references.md already says the regime
was solved in 2007), and the general "weak or missing baselines" claim is McGreivy & Hakim. "Layout randomisation breaks a
fixed linear map" is close to true by construction (layout changes the operator), so the contribution is the
measurement and its use as a benchmark-design diagnostic, not the insight. No paper found reporting ridge-versus-layout
on chip thermal data. Unchanged from the earlier check: our public README still carries stale numbers (ridge R^2 about
0.99, eight geometries) and should be fixed before any public claim.

## Finding 4. Independent-discretisation differential testing as a bug-finding tool

| Work | Overlap | Provenance |
|---|---|---|
| Code-to-code comparison practice (e.g. Ahmad, NASA NAS 2011, [pdf](https://www.nas.nasa.gov/assets/nas/pdf/papers/ahmad_j_code_to_code_comparison_2011.pdf); Santa Barbara cluster comparison) | Comparing independent codes to find bugs and explain discrepancies is standard verification practice in CFD, astrophysics, HED physics. | snippet |
| Chen, Feng & Tse (2002), "Metamorphic testing of programs on partial differential equations: a case study", COMPSAC 2002, [link](https://www.cs.hku.hk/~tse/Papers/2000s/jqpdeCX.html) | Metamorphic testing finds errors in an elliptic-PDE program without an oracle. | snippet |
| Kanewala & Bieman, "Testing Scientific Software: A Systematic Literature Review", [arXiv:1804.01954](https://arxiv.org/abs/1804.01954); Segura et al. 2016 TSE survey | Surveys metamorphic and differential techniques for scientific software. | snippet |
| ATSim3.5D vs Icepak/HotSpot; 3D-ICE 4.0 vs HotSpot, COMSOL, PACT | Thermal simulators compared as accuracy benchmarks. | snippet / abstract |
| HotSpot 3.0 grid-model bug revealed by block-vs-grid disagreement ([HotSpot versions page](https://lava.cs.virginia.edu/hotspot/versions.htm)) | A thermal-simulator bug exposed by internal cross-model disagreement. | snippet, page not opened |

**Verdict: PARTIALLY ANTICIPATED, and the method itself is ANTICIPATED.** Differential testing and code-to-code
comparison are standard. Within chip thermal, the HotSpot version history is a precedent of a bug found by model
disagreement. What remains ours is the packaged instance: an independent exact-area-fraction FV reference, refinement-based
tolerance, a classifier for "explained" residuals (the 3D-ICE node-placement convention), and a pre-registered test on
archived real data (23/24 and 24/24). Call it a tool and an application, never a new technique.

## Finding 5. Linter / sanity-checker for PDE-surrogate datasets

| Work | Overlap | Provenance |
|---|---|---|
| Shikhman (2026), diagnostic suite for learned PDE simulators, [arXiv:2606.18200](https://arxiv.org/abs/2606.18200) | Audits trained models (semigroup, energy, balance, admissibility), not datasets. | found |
| Li et al. (2026), Domain-Validity-Gated Metamorphic Testing of Scientific ML Surrogates, [arXiv:2606.17529](https://arxiv.org/abs/2606.17529) | Metamorphic tests of surrogates (permutation, mirror, conservation); model-side. | found |
| Roohi (2026), FlowMLLab, [arXiv:2609.27975](https://arxiv.org/abs/2609.27975) | CFD-to-ML pipeline with case-wise leakage-proof splits, matched non-neural baselines, physical diagnostics. A framework, not a dataset linter; abstract only. | found |
| Jakeman, Barba, Martins & O'Leary-Roseberry (2026), "Verification and validation for trustworthy scientific machine learning", [IOP](https://iopscience.iop.org/article/10.1088/2632-2153/ae59ec) | 16 recommendations incl. documenting data fidelity and processing; no tool. | found |
| Northcutt et al. (2021), [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) | Label-error detection in general ML benchmarks. Not PDE-aware. | found |

**Verdict: NOVEL for the specific tool, with a weak-evidence caveat.** Everything found is model-side or a general
data-quality framework. None checks orientation, maximum principle, input/label pairing, or operator residuals on
simulator-generated datasets. The search tool does not index GitHub or PyPI well, so an unpublished repository could
exist. The fieldlint round-1 held-out failure (frozen v1 failed on Darcy) means any claim must be about v2 and its
own held-out set only.

## Finding 6. IC-ThermBench transposed input channels

| Source | Finding | Provenance |
|---|---|---|
| [arXiv:2608.23977](https://arxiv.org/abs/2608.23977) | v1 2026-08-25, v2 2026-09-07; no later version or erratum visible. | found |
| [github.com/Day333/ThermalBench](https://github.com/Day333/ThermalBench) | Issue tracker shows no issues. README documents tensor `(B, X, Y, Z, P)` and says S1 data are released but its unified evaluation script is still in progress. | found |
| Web search for erratum or orientation-bug reports | None. | found |

**Verdict: NOVEL (no report found).** Evidence is a negative on a small tracker as of 2026-10-07. Report it upstream:
it is a clean, verifiable finding and costs little.

## Overall verdict

| # | Finding | Verdict | Closest work |
|---|---|---|---|
| 1 | Surrogate reproduces solver artefact, paired projection test | PARTIALLY ANTICIPATED | Mohan et al., arXiv:2411.15101 |
| 2 | `.lyt` partial-cell inconsistency | NOVEL (as a report; soften "internal inconsistency") | 3D-ICE 4.0, arXiv:2512.05823 |
| 3 | Ridge solves fixed-placement; layout randomisation breaks it | PARTIALLY ANTICIPATED | McGreivy & Hakim, arXiv:2407.07218 |
| 4 | Differential testing with an independent discretisation | PARTIALLY ANTICIPATED (method anticipated) | Code-to-code comparison; Chen, Feng & Tse 2002 |
| 5 | PDE-dataset linter | NOVEL (limited search reach) | Shikhman, arXiv:2606.18200 (model-side) |
| 6 | IC-ThermBench transpose | NOVEL (no report found) | none; ThermalBench tracker empty |

Defensible claim set: a measured, mechanistic, pre-registered demonstration (Finding 1 test, Finding 2 bug, Finding 3
numbers), plus two open tools. Not defensible: any claim that surrogates learning solver artefacts, linear-baseline
critique, or differential testing is a new idea. Cheapest next steps: read the full text of Mohan et al. and
arXiv:2608.24547 to confirm neither contains a paired test; report Findings 2 and 6 upstream.
