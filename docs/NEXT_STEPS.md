# Next steps

Everything so far is **synthetic**. Nothing here shows SteadyVoice works on real speech-BCI data. Order matters: do 1-3 before anything else.

## 1. Get the real data (blocked on a human)
- Dryad doi:10.5061/dryad.dncjsxm85 refused scripted downloads from the build container (403 web / 401 API).
- Either download `t15_copyTask_neuralData.zip` (11 GB) in a browser, or create a Dryad API token and run
  `DRYAD_TOKEN=... scripts/download_t15.sh <dest>` (written but never tested with a real token).
- Unzip, then `python scripts/inspect_hdf5.py <root>`.

## 2. Verify and fix the loader
`steadyvoice/data/t15.py` was written from memory of the public baseline repo and is only tested against a fake file with the assumed schema
(`<session>/data_{train,val}.hdf5`, `input_features` [T,512], `seq_class_ids`, `seq_len`, session names `t15.YYYY.MM.DD`).
Compare against the inspector output; fix mismatches. Check: 41 classes with 0 = blank and 40 = SIL; whether features need per-block z-scoring
(the baseline normalises; the loader does not); that some sessions have no val split.

## 3. Sanity-check the pipeline on real data (before any drift claim)
- Train the backbone on all sessions' train splits briefly and confirm greedy val PER is in the ballpark of the published ~10 % (it will be worse with a causal GRU and no LM; the point is "not broken").
- The released all-session baseline weights may be used for this sanity check ONLY. They saw every session's labels, so they must never be used for a drift curve.

## 4. Run the actual experiment on the RTX 3050
`python scripts/run_real.py --root <root>/hdf5_data_final --k 5` (hidden 512 / 4 layers / AMP by default; if it OOMs lower `--bs` or `--hidden`).
Then k = 3 and 10, 3 seeds each. Expect ~8-12 h per backbone run; the forward simulation itself is cheap.
First check to make: does the no-adaptation PER really rise with days since training? If not, the premise does not hold for this participant.

## 5. What must be re-examined on real data (the synthetic results do not transfer automatically)
- **Guard predictor.** Trained on gain/offset/rotation/dead-channel perturbations; real drift may differ. The natural cross-day pairs are the assumption-free part. The reliability gate (fit rho < 0.5 -> scheduled calibration fallback) exists because a natural-only fit can be degenerate. Report Spearman(signal, true PER) on held-out days.
- **Language model.** The synthetic LM is a phoneme n-gram trained on the same grammar. For real data, build a word-level KenLM from external text with benchmark sentences removed (leakage), or at minimum a phoneme n-gram from training-session text only. Always report LM-off PER as primary.
- **Adapter size.** 512x512 affine on a few dozen unlabelled sentences may overfit; try low-rank / diagonal (already implemented) and keep the proximal anchor.
- **CORAL warm start.** It hurt on synthetic rotation drift and helped some seeds; ablate it on real data.

## 6. Known limitations to fix or state in any write-up
- Only 5-10 synthetic seeds; wide CIs. Component ablations were confounded by collapse days.
- Guard hyperparameters (margin, tau_cal) were tuned on synthetic dev seeds only.
- The fallback gate was designed after looking at seeds 0-4; only seeds 5-9 evaluate it.
- Single (future) participant: claims are about T15's drift, not drift in general.
- Offline results only; no clinical claims. Follow the dataset licence and credit the participants and the UC Davis team.

## 7. Extensions (only after 1-4)
B2T '24 / T12 replication as a second participant; causal-transformer backbone; full-network fine-tune ablation (CORP original);
cross-participant warm start of the adapter; non-invasive (LibriBrain) transfer; imagined speech.
