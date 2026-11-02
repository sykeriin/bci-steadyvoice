# SteadyVoice — label-free day-to-day recalibration for speech BCIs

Status: **pipeline built and verified on synthetic drifting data. Not yet run on real T15 data.**
Full plan, rationale and rejected alternatives: [docs/PLAN.md](docs/PLAN.md).

## What is here
- Causal GRU-CTC backbone; only a small per-day input adapter adapts (identity-initialised).
- Baselines: none, oracle (daily labelled refit), periodic labelled refit, CORAL, Tent, CORP-style, guard-only.
- SteadyVoice: CORAL warm start + LM-gated pseudo-labels + latent alignment + augmentation consistency + proximal anchor,
  and a collapse guard (label-free signals -> predicted PER fitted on *training days only* -> accept / reject / rollback / request calibration).
- Leak-proof forward simulation: methods get features only; labels are reachable only via a counted `CalibrationService`.
- `pytest tests` (12 tests): causality, identity adapters, LM normalisation, CORAL math, guard logic, and leakage
  (corrupting all future days and all labels leaves day-d state bit-identical).

## Synthetic results (5 seeds; LM-off PER; k=5 training days, 21 forward days)
Each seed = new language, drift trajectory and backbone. mean ± std across seeds.

| method | mean PER | labels used (of 420 for daily oracle) |
|---|---|---|
| none | 0.569 ± 0.157 | 0 |
| corp | 0.551 ± 0.319 | 0 |
| steadyvoice_noguard | 0.222 ± 0.178 | 0 |
| periodic7 (fixed schedule) | 0.125 ± 0.051 | 80 |
| guard_only (no label-free adaptation) | 0.089 ± 0.014 | 68 ± 16 |
| **steadyvoice (guard)** | **0.036 ± 0.020** | **28 ± 20** |
| oracle (daily calibration) | 0.012 ± 0.005 | 420 |

Full table incl. ablations: `results/synthetic/summary.md`; per-seed curves: `results/synthetic/per_vs_days_*.png`.

What this supports:
- Label-free pseudo-labelling (CORP) is excellent on gradual drift (0.8% PER on moderate days) but **collapses on jumps**
  (worse than doing nothing in 3 of 5 seeds). Confidence gating alone stops the spiral but cannot recover.
- Guarded SteadyVoice beat the label-matched baselines (periodic7, guard_only) in 5/5 seeds, using ~15x fewer labels than daily calibration.
- It does not match daily calibration (3.6% vs 1.2%).
- The guard costs accuracy when drift is benign (it rejects some good updates).

What it does NOT support (be skeptical):
- **Synthetic only.** Drift here is gain/offset/rotation/dead channels/jumps that I designed; the noise and drift strength were tuned
  so that no-adapt PER visibly decays. Real T15 drift may differ completely.
- The guard's predictor is trained on perturbations from the same family as the synthetic drift (well-specified). On real data this
  is the main risk; natural cross-session pairs from training days are the assumption-free part.
- 5 seeds, large variance. Component ablations (sv_no*) are confounded by collapse days and I would not rank components from them.
- "Moderate/severe" regimes were defined post-hoc (unadapted PER < / >= 0.30). The `collapse_events` metric (>2x worse than no adaptation)
  is nearly useless here because no-adaptation is itself bad on those days; use worst-day PER instead.
- Hyperparameters were set a priori, not tuned, but only on this synthetic world.
- The LM is a phoneme n-gram (not word-level KenLM) trained on the synthetic grammar.

## Run
```bash
pip install -r requirements.txt
pytest tests -q
python scripts/run_synthetic.py --seed 0                    # ~7 min on 4 CPUs
```
### Real data (RTX 3050) — UNTESTED
1. Download `t15_copyTask_neuralData.zip` from Dryad (doi:10.5061/dryad.dncjsxm85; needs a browser or API token) and unzip.
2. `python scripts/inspect_hdf5.py <root>` — check the schema matches `steadyvoice/data/t15.py` (written from memory, unverified).
3. `python scripts/run_real.py --root <root>/hdf5_data_final --k 5` (retrains the backbone on the first k sessions only;
   never use the released all-session weights for drift curves: they saw every session's labels).
