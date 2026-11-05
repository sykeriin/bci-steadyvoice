# SteadyVoice — label-free day-to-day recalibration for speech BCIs

**Status: pipeline built, tested (18 tests) and evaluated on SYNTHETIC drifting data only. Nothing has been run on real T15 data.**
Real data was blocked (Dryad refused scripted downloads); see [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md). Plan, rationale and rejected alternatives: [docs/PLAN.md](docs/PLAN.md).

## What is here
- Causal GRU-CTC backbone; only a small per-day input adapter adapts (identity-initialised).
- Baselines: none, oracle (daily labelled refit), periodic labelled refit, CORAL, Tent, CORP-style, guard-only (guard + calibration, no adaptation).
- SteadyVoice: CORAL warm start + LM-gated pseudo-labels + latent alignment + augmentation consistency + proximal anchor, and a collapse guard
  (label-free signals -> predicted PER fitted on training days only -> accept / reject / rollback / request a 20-sentence calibration).
- Leak-proof forward simulation: methods receive features only; labels are reachable solely through a counted `CalibrationService`.
- Tests: causality, identity adapters, LM normalisation, CORAL math, guard logic and fallback, day-gap isolation, probe, loader, and leakage
  (corrupting all future days and all labels leaves day-d state bit-identical).

## Main result: 10 synthetic seeds, LM-off PER, k=5 training days, 21 forward days
Guard parameters tuned on separate dev seeds (100-102), then frozen. Each seed = new language, drift trajectory and backbone.

| method | mean PER | worst-day PER | labels used (daily calibration = 420) | days > 2x oracle |
|---|---|---|---|---|
| none | 0.526 ± 0.140 | 0.90 | 0 | 16.3 |
| CORP-style | 0.478 ± 0.288 | 0.80 | 0 | 10.2 |
| SteadyVoice, no guard | 0.264 ± 0.171 | 0.65 | 0 | 8.5 |
| periodic labelled refit (every 7 days) | 0.125 ± 0.063 | 0.71 | 78 | 9.3 |
| guard + calibration, no adaptation | 0.066 ± 0.013 | 0.19 | 76 ± 15 | 10.3 |
| **SteadyVoice (guard)** | **0.024 ± 0.008** | **0.087** | **24 ± 22** | **1.1** |
| oracle (daily calibration) | 0.011 ± 0.004 | 0.04 | 420 | 0 |

Guarded SteadyVoice beat every label-spending baseline in 10/10 seeds (vs guard-only: mean diff 0.042, bootstrap 95% CI [0.034, 0.051]; vs periodic: 0.101 [0.066, 0.138]).
It does not match daily calibration (0.024 vs 0.011). Full tables: `results/final/summary.md`, `results/final/gap_probe.md`.

What the evidence says:
- Label-free pseudo-labelling (CORP) is excellent on gradual drift (0.7% PER on moderate days) but **collapses on jumps**: worse than no adaptation in several seeds.
- Gating alone stops the spiral but cannot recover; the guard's calibration request is what makes it recover.
- After a calibration, label-free adaptation keeps PER low for many days. That is its contribution beyond the guard (0.066 -> 0.024 vs guard-only).
- **Linear probe:** adapted latents keep phoneme information (probe PER 0.026 vs frozen-head 0.024), so the adapter is not silently destroying signal.

## Where it is weak (read this before believing the table)
1. **Synthetic only.** I designed the drift (gain/offset/rotation/dead channels/jumps) and tuned its strength so the unadapted decoder visibly decays. Real drift may differ completely.
2. **Long gaps hurt SteadyVoice.** Adapt once after an unobserved gap (PER, labels in brackets): g=5 0.021 [4] vs guard-only 0.031; **g=10 0.076 vs guard-only 0.041; g=20 0.178 vs guard-only 0.056**.
   Label-free adaptation cannot recover from a long gap, and the guard's calibration then starts from a worse state or the predictor under-estimates PER. Not fixed. (periodic7 equals "none" in this test by construction.)
3. **Guard predictor risk.** Trained on perturbations from the same family as the synthetic drift. Unseen smooth nonlinear drift did not hurt it (5 seeds: 0.030 vs 0.036) but that is a weak test.
   Fit on natural cross-day pairs only it was worse on average (0.143 ± 0.211, seed 0 degenerate: every training pair had ~0 PER so the guard never fired).
4. **The reliability fallback is not shown to help.** Added after seeing that seed-0 failure (predictor fit rho < 0.5 -> scheduled 7-day calibration instead of label-free adaptation).
   On fresh seeds 5-9 (natural-only predictor), fallback ON scored 0.031 vs OFF 0.021: it fired on seed 5 (0.079 vs 0.028) where the predictor would have worked. The failure it guards against did not recur in these seeds,
   so its benefit is untested, and the 0.5 threshold looks too aggressive (a NaN-only gate would target the degenerate case). Left as is, as insurance with a measured cost.
5. **Tuning gain is small.** On the 3 dev seeds the best setting (margin 0, tau_cal 0.2) beat the old defaults by ~0.003 PER, within noise; the apparent gain on seeds 0-4 (0.036 -> 0.026) is mostly seed variance and tuning should not be credited for it.
6. Component ablations (sv_no*) were run on 5 seeds, are confounded by collapse days, and I would not rank components from them.
7. LM is a phoneme n-gram trained on the synthetic grammar (not word-level KenLM). `collapse_events` is a weak metric here; use worst-day PER and days > 2x oracle.
8. Moderate/severe regimes were defined post-hoc (unadapted PER < / >= 0.30). Synthetic dev results are not evidence about clinical value; no clinical claims.

## Run
```bash
pip install -r requirements.txt
pytest tests -q
python scripts/run_final.py --seeds 0            # ~7 min per seed on 4 CPUs; resumable
python scripts/aggregate.py results/final        # tables with paired bootstrap CIs
python scripts/gap_probe_summary.py results/final
```
Real data (RTX 3050): follow [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md). `steadyvoice/data/t15.py` is **unverified** against the real HDF5 schema; never use the released all-session weights for drift curves (they saw every session's labels).
