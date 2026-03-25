# self-replication phase diagram

what substrate features allow life?

## question

which combinations of cellular automata properties permit self-replicating structures to emerge? we exhaustively classify all 262,144 outer-totalistic binary CA rules with Moore neighbourhood for self-replication and produce phase diagrams in the (lambda, F) plane.

## findings

1. self-replication concentrates at low lambda (~0.15-0.25) and low-to-moderate F (~0.2-0.3), in the weakly supercritical regime (Derrida mu = 1.81 for replicators vs 1.39 for non-replicators)
2. 20,152 of 262,144 Life-like rules (7.69%) support pattern proliferation
3. approximate mass conservation is the dominant predictor (Cohen's d = -0.94, logistic AUC = 0.85); generalises to k=3 (d = -1.04)
4. O-information synergy is a real univariate signal (d = -0.27) but adds no independent power beyond mass-balance
5. under equalised detection protocol, self-replication rate increases monotonically with neighbourhood size: vN 4.79% < Moore 7.69% < extended Moore 16.69%
6. 97.8% of tier-1 rules confirmed at extended simulation length; 20.8% pass causal perturbation test; estimated 1.56% causal self-replication rate
7. k=3 rate is double k=2 (15.55% vs 7.69%); C4-symmetric rules are lower (3.57%)
8. ASAL visual open-endedness shows zero correlation with self-replication (Spearman rho = -0.002)

## directory structure

```
run.py                       # single entry point, 25 pipeline stages
environment.yml              # micromamba env spec (exp04-selfrep)
docs/
  PROJECT.md                 # this file
  SUBMISSION-PLAN.md         # venue strategy and timeline
src/
  config.py                  # paths, constants, NEIGHBORHOOD_OFFSETS
  modules/
    fast_sim.py              # FFT-based CA simulator (118ms/rule)
    general_sim.py           # general binary CA for non-totalistic rules
    simulator.py             # numba reference simulator
    rule_params.py           # lambda, F computation, rule enumeration
    detector.py              # 3-stage detection: screening, matching, causal test
    measures.py              # mass-balance, O-information, spatial entropy
  utils/
    plots/
      __init__.py            # LaTeX style config (newtx fonts, 300 DPI)
      scatter_phase.py       # discrete grid phase diagram
      surface3d.py           # smoothed 3D surface with density color
      boundary.py            # boundary measure scatter plots
jobs/
  validate_pipeline.py       # GoL/HighLife/dead/chaotic benchmarks
  k2_census.py               # k=2 vN exhaustive (1024 rules)
  k2_moore_census.py         # k=2 Moore exhaustive (262K rules)
  boundary_measures.py       # mass-balance + spatial entropy (300+300)
  oinfo_boundary.py          # O-information (300+300)
  tier2_check.py             # extended-length rescreen (1000 rules)
  tier3_causal.py            # causal perturbation test (978 rules)
  k3_sweep.py                # k=3 Moore sample (10K rules)
  k2_c4_sample.py            # C4 rotationally-symmetric (10K rules)
  extended_moore_sweep.py    # k=2 extended Moore |N|=25 (10K rules)
  k2_vn_equalized.py         # k=2 vN equalised protocol (1024 rules)
  derrida_phase_diagram.py   # Derrida mu across (lambda, F) (~1000 rules)
  f_ablation.py              # F-weighting robustness (uniform/linear/quadratic)
  mass_balance_cross_substrate.py  # mass-balance for k=3 + vN
  asal_comparison.py         # ASAL open-endedness cross-comparison
  matched_controls.py        # lambda/F-matched negatives + permutation
  effect_sizes.py            # Cohen's d, rank-biserial for all measures
  f_marginal.py              # replication rate vs F
  logistic_classifier.py     # logistic regression + ROC
  sensitivity_sweep.py       # detection-threshold sensitivity
  verification_run.py        # 256x256 / 8192 step verification
  replicator_viz.py          # time-lapse of example replicators
  figures.py                 # all publication figures
  new_figures.py             # Derrida, F-ablation, cross-substrate figures
  tier_examples_viz.py       # tier detection examples figure
```

## pipeline

```bash
micromamba run -n exp04-selfrep python run.py
```

25 stages, all idempotent (skip if artifact exists). total runtime ~2.5 hours on a single workstation.

| phase | stages | compute |
|---|---|---|
| A: validation | validate_pipeline | seconds |
| B: exhaustive censuses | k2_census, k2_moore_census | 8.7h (Moore, already cached) |
| C: boundary analysis | boundary_measures, oinfo_boundary, tier2_check | ~30 min |
| D: extensions | k3_sweep, k2_c4_sample, tier3_causal, extended_moore_sweep | ~30 min |
| D2: new analyses | k2_vn_equalized, derrida_phase_diagram, f_ablation, mass_balance_cross_substrate, asal_comparison, sensitivity_sweep | ~15 min |
| E: statistical controls | matched_controls, effect_sizes, f_marginal, logistic_classifier | seconds |
| F: verification + figures | verification_run, replicator_viz, figures, new_figures, tier_examples_viz | ~5 min |

## how to run

```bash
micromamba create -f environment.yml -y
micromamba run -n exp04-selfrep python run.py
```

## dependencies

see `environment.yml`. requires: numpy, scipy, numba, matplotlib, scikit-learn.
