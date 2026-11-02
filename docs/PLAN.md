# SteadyVoice: label-free day-to-day recalibration for speech BCIs (plan)

Date: 2026-11-02 · Owner: DK · Target end: about 2027-01-15 (10 weeks part-time, minus the holiday week)

---

## 0. Context

**The problem:** intracortical speech BCIs (Willett 2023, Card 2024) decode at conversational rates, but the mapping from neurons to phonemes drifts between sessions. Today each day starts with a labelled recalibration block. Label-free self-recalibration with LM pseudo-labels (CORP) works when drift is small. When drift is large it feeds its own mistakes back in and collapses.

**What we build:** a decoder whose only adaptable part is a small per-day input adapter. It is updated each day from unlabelled data using three signals, and a collapse guard can veto, roll back, or ask for a short calibration. It is evaluated as error against days since the last labelled calibration, with the LM switched off as the primary number.

**Why this is worth doing:** the gap is not accuracy. It is a decoder that (a) needs no labels, (b) knows when it is failing, (c) is causal, and (d) is measured over time rather than on a single held-out split.

### Honest check on the project brief (things I'm changing)
1. **Citation error:** the brief attributes CORP to arXiv 2507.02800. That ID is the Time-Masked Transformer + TTA paper (Feghhi/Kao). CORP is **Fan et al., NeurIPS 2023, arXiv 2311.03611**, "Plug-and-Play Stability for Intracortical BCIs: A One-Year Demonstration…". I checked both IDs on arXiv.
2. **Leakage in Week 1 as written:** the released B2T '25 RNN weights were trained on all 45 sessions with labels. Using them for the drift curve means the backbone has already seen every "future" day. They are good only for a reproduction sanity check. **Every drift result must come from a backbone retrained on the first k sessions only.** This is the main extra compute cost.
3. **Channel count:** T15 has 256 electrodes but **512 features** (threshold crossings + spike-band power, 20 ms bins). The adapter works on 512 dimensions.
4. **The 5-gram + OPT pipeline won't run here or on a 3050.** The released 5-gram LM needs roughly 300 GB of RAM, and OPT-6.7B doesn't fit in 3050 VRAM. We train our own KenLM 3/4-gram on a corpus **with the benchmark sentences removed**, which also handles the LM-leakage concern.
5. **Novelty is moderate. Be realistic.** LM pseudo-labels (CORP), adversarial alignment (ALIGN), and collapse-resistant TTA with resets (SAR 2302.12400, RDumb 2306.05401, RDumb++ 2601.15544) all exist. The new parts are: (i) a collapse guard whose unsupervised signals are **shown to predict true PER** on held-out days, (ii) a leak-proof forward-in-time protocol with LM-off reporting, and (iii) the combination of the two in a causal, adapter-only setting. That is realistically a strong workshop paper or an IEEE NER / JNE-level submission, not a Nature paper.

### Constraints confirmed in this session
- Cloud container: 4 CPU, 15 GB RAM, no GPU, 31 GB of free disk. **Your GPU: RTX 3050** (assume 4–8 GB VRAM, so use AMP and keep hidden size ≤ 768).
- Dryad dataset `doi:10.5061/dryad.dncjsxm85` is reachable: `t15_copyTask_neuralData.zip` 11.05 GB, `t15_pretrained_rnn_baseline.zip` 0.48 GB, `t15_copyTask.pkl` 0.06 GB. The file API returned "Unauthorized" for the README, so **the download may need a Dryad token**. Fallback: you add a token or Kaggle key as an environment secret.
- GitHub `Neuroprosthetics-Lab/nejm-brain-to-text` (the baseline code) isn't attached. I'll request read access via add_repo to mirror its data format and model, not to copy it wholesale.
- **Needed from you:** the GitHub `owner/repo` to push to.

---

## 1. Scope

**In scope (v1)**
- T15 (B2T '25) as primary data. A causal 5-layer GRU CTC phoneme decoder trained on sessions 1..k.
- Per-day adapter variants: full affine 512×512 (identity init), low-rank (r = 16/32), diagonal + bias.
- Adaptation methods: none, oracle-supervised, CORP-style, CORAL/Procrustes alignment, Tent (negative control), SteadyVoice.
- Collapse guard with rollback and a counted labelled-calibration fallback.
- Forward simulation for k ∈ {3, 5, 10}, plus a day-gap stress test (skip sessions).
- Metrics: PER (LM-off, primary), WER with our n-gram, collapse events, rollback rate, labels saved, real-time factor.

**Stretch (v1.5):** B2T '24 / T12 replication (different participant, 128 electrodes, a second check of generality). FALCON sanity check.

**Out of scope:** LLM rescoring, BIT backbone (weights and VRAM are uncertain), MEG/LibriBrain, imagined speech, any clinical claim.

---

## 2. System design

```
steadyvoice/
  data/
    t15.py          # HDF5 loader → Trial(features[T,512], phonemes, text, session_id, date, split)
    stream.py       # SessionStream: yields DayBatch(unlabelled) in date order; labels held privately
    synthetic.py    # Synthetic drifting data: gain/offset/channel loss/rotation of latent → testable w/o real data
  models/
    backbone.py     # Causal GRU (unidirectional), patching (size 14, stride 4), CTC head (41 classes incl. blank)
    adapters.py     # AffineAdapter, LowRankAdapter, DiagAdapter; all identity-initialised, serialisable
  decode/
    ctc.py          # greedy + beam search (torchaudio ctc_decoder w/ lexicon + KenLM)
    lm/             # build_lm.py: KenLM n-gram from corpus with benchmark sentences filtered out
    g2p.py          # CMUdict word→phoneme for mapping LM text back to phoneme pseudo-labels
  adapt/
    base.py         # interface: adapt(model, prev_state, day_unlabelled) -> (new_state, diagnostics)
    none.py, oracle.py, corp.py, coral.py, tent.py, steadyvoice.py
  guard/
    signals.py      # entropy, blank ratio, LM-share, adapted-vs-frozen divergence, CORAL distance
    policy.py       # accept / reject-update / rollback / request-calibration; thresholds fit on train days only
    store.py        # versioned adapter checkpoints (last-good pointer)
  eval/
    runner.py       # forward simulation; the only object that sees labels (for scoring after adapt())
    metrics.py      # PER, WER, collapse events, rollback rate, labels used, RTF
    plots.py        # headline: PER/WER vs days-since-calibration, 5+ curves, shaded seeds
  configs/*.yaml    # plain dataclasses + yaml (no Hydra)
scripts/            # download_data.sh, train_backbone.py, run_forward_sim.py, build_lm.py
tests/              # leakage test, adapter-identity test, synthetic-drift recovery test, guard rollback test
```

**Leakage-proof by construction:** `SessionStream` hands `adapt()` a `DayBatch` with features only. Labels live in a separate `LabelVault` that only `eval.runner` can read, and only *after* `adapt()` returns for that day. A unit test checks that no adaptation method can import the vault, and that the day-d state depends only on days ≤ d. Within a session we adapt on that session's train + test trials (unlabelled) and score on its val trials. The B2T '25 test split has hidden labels anyway.

**Per-day loop (SteadyVoice):**
1. Initialise today's adapter from the last-good adapter (chain), with a proximal anchor to it.
2. Closed-form warm start: CORAL / Procrustes on latent statistics against reference days.
3. Gradient steps on the adapter only, using
   `L = λ_pl·CTC(gated pseudo-labels) + λ_al·CORAL(latent) + λ_cons·KL(weak‖strong aug) + λ_prox·‖W − W_prev‖²`.
   - Gated pseudo-labels: greedy decode, then LM beam decode, then G2P back to phonemes. Keep a trial only if 1 − normEditDist(greedy, LM) ≥ τ and mean frame confidence ≥ c.
   - Augmentations: channel dropout (10–20%), ±1–2 bin shifts, small gain noise.
4. Guard computes signals on today's unlabelled data for the frozen adapter, the previous adapter, and the candidate adapter, then decides:
   accept → commit · reject → keep previous · rollback → last-good · **request calibration** → supervised refit of the adapter on N = 20 labelled sentences (counted as labels used).

**Guard calibration (the research core):** fit thresholds or a tiny logistic model ("PER > 1.5× reference?") on **training days only**, using synthetic perturbations of real data (gain drift, channel death, offsets, rotated latents) as labelled examples of failure. On future days, report the Spearman correlation between each guard signal and true PER, and the guard's precision/recall for collapse. If the signals don't predict PER, that is a publishable negative result and we report it as one.

---

## 3. Evaluation protocol
- Train the backbone (with per-day adapters for training days) on sessions 1..k with labels. Seeds: 3.
- For d = k+1 … 45 in date order: `adapt()` without labels, then score val trials. The x-axis is **days since last labelled calibration** (real calendar days, from session dates), not session index.
- Curves: no-adapt, CORP-style, CORAL/ALIGN-lite, Tent, SteadyVoice (with guard and with guard off), oracle (labelled refit each day).
- **Day-gap stress:** adapt directly from the session-k state to sessions k+5, k+10, k+20.
- Ablations: each of the three signals removed; adapter type; λ_prox = 0; LM on/off; k.
- **Collapse event:** PER on day d > 2× the no-adapt PER on day d, or > 2× the previous day. The brief's "WER jumps 2×" is noisier with an LM in the loop.
- Sanity reproduction: run the released all-session weights on val with greedy CTC. PER should land near the published ~10%. This checks the loader and model, nothing else.

---

## 4. Timeline (actual calendar)
| Week | Dates | Deliverable |
|---|---|---|
| 1 | Nov 2–8 | Repo, loader, synthetic data, causal GRU, adapters, stream/vault, tests (built in this container, CPU). Data download. Released-weights PER sanity check (CPU inference) |
| 2 | Nov 9–15 | `train_backbone.py` on your 3050 for k = 5. No-adapt decay curve. KenLM build with sentence filtering |
| 3–4 | Nov 16–29 | Alignment adapters (CORAL, Procrustes), oracle, Tent baseline |
| 5 | Nov 30–Dec 6 | CORP-style baseline + gated pseudo-labels |
| 6 | Dec 7–13 | Augmentation consistency + combined loss |
| 7 | Dec 14–20 | Guard: signals, calibration on train days, rollback, day-gap stress |
| — | Dec 21–Jan 3 | Holiday buffer. Background runs for k = 3/10, seeds |
| 8 | Jan 4–10 | LM-off ablations, B2T '24 replication if time allows |
| 9–10 | Jan 11–24 | Streaming RTF on CPU, linear-probe check, write-up, code release |

Compute estimate: backbone of about 33M params, around 8–12 h per training run on a 3050 with AMP. That's 3 k values × 3 seeds = 9 runs, about 4–5 GPU-days. If that's too slow, drop to hidden size 512 (about 3× faster), report it as such, and keep comparisons within the same backbone.

---

## 5. Decisions and rejected alternatives
| Decision | Chosen | Rejected and why |
|---|---|---|
| What adapts | Input adapter only | Full-network fine-tune (CORP adapts the GRU): more capacity to collapse and harder to roll back. Kept as an ablation |
| Backbone | Causal GRU | BiGRU (top B2T '24 entries): can't stream. Causal transformer (2507.02800): better, but more VRAM, and a second variable. BIT: weights and size uncertain |
| Alignment | CORAL/Procrustes | ALIGN adversarial: needs pooled sessions and GAN instability; used only as an "ALIGN-lite" baseline. NoMAD/LFADS dynamics alignment: heavy, needs a generative model per participant |
| Pseudo-labels | LM-gated agreement | Raw LM outputs (CORP): this is the failure mode under study. Entropy minimisation (Tent): collapses toward blank in CTC, kept as a negative control |
| LM | Own KenLM n-gram, benchmark sentences filtered | Released 5-gram (about 300 GB RAM). OPT/LLM rescoring: VRAM, latency, and it hides neural decay |
| Guard | Explicit signals + rollback + counted labelled fallback | Periodic blind reset (RDumb): simple, a good baseline, but throws away good adaptation. No guard: unsafe, since it puts words in a user's mouth |
| Primary metric | LM-off PER vs calendar days | Single held-out WER: hides drift and LM leakage |
| Framework | PyTorch + plain yaml | JAX (the baseline is PyTorch). Hydra (overkill) |
| Released weights | Sanity check only | Using them for drift curves leaks future-day labels |

## 6. Related work (reading order)
Willett 2023 Nature · Card 2024 NEJM (T15/Dryad) · B2T '24 lessons (2412.17227) · **CORP, Fan 2023 (2311.03611)** · Time-masked transformers + TTA (2507.02800) · ALIGN (2603.18299) · BIT (2511.21740) · Survey (2603.12279) · LightBeam CTC decoder (2603.14002) · Degenhart 2020 Nat BME (manifold alignment stabilisation) · Karpowicz 2022 NoMAD · Farshchian 2019 ADAN · Wilson et al. unsupervised cursor-BCI recalibration (Stanford; likely related to docket S22-226) · CORAL, Sun & Saenko 2016 · FixMatch, Sohn 2020 · Tent, Wang 2021 · SAR (2302.12400) · RDumb (2306.05401) / RDumb++ (2601.15544) · LibriBrain/PNPL (2506.10165).

## 7. Risks
- The Dryad download needs auth. Mitigation: synthetic pipeline first, then a token or Kaggle key.
- Single participant: claims are limited to "T15's drift".
- Guard signals don't predict PER: report it, and fall back to RDumb-style periodic calibration as the safe policy.
- Session dates are unevenly spaced: bin by calendar days and show per-session points.
- 3050 VRAM: AMP, gradient accumulation, hidden size 512 fallback.
- Disk: 11 GB zip + extraction ≈ 22 GB of the 31 GB free. Delete the zip after extracting.
- Ethics: follow the Dryad licence (CC0), credit T15 and the UC Davis team, and make no clinical claims from offline results.

## 8. Build order in this session (what I'll do after approval)
1. Ask for the repo name, add_repo it, scaffold the package above, and add `requirements.txt` (torch CPU, h5py, numpy, scipy, torchaudio, kenlm, jiwer, matplotlib, pytest).
2. `synthetic.py` + all model/adapter/adapt/guard/eval code, tested end to end on synthetic drift (the "none" PER must rise over days, oracle must stay flat, and the guard must roll back on an injected collapse).
3. Try the Dryad download. If it works, inspect the HDF5 schema, write `t15.py`, and run the released-weights greedy-PER sanity check on CPU.
4. Write `scripts/train_backbone.py` + `run_forward_sim.py` with a README section "Run on your 3050".
5. Commit and push to your branch.

## 9. Verification
- `pytest tests/`: leakage test, adapter identity, synthetic recovery, guard rollback.
- `python scripts/run_forward_sim.py --data synthetic --methods all` produces the headline plot with the expected orderings.
- On real data: the released-weights greedy PER is close to the published value (confirms the loader). After your 3050 run, the no-adapt curve should degrade with days since calibration.
