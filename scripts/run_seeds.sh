#!/usr/bin/env bash
# Sequentially run the synthetic experiment over seeds (each seed = new language, drift trajectory, backbone).
set -e
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do
  python scripts/run_synthetic.py --seed $s --days 26 --out results/synthetic \
    --methods none,oracle,periodic7,coral,tent,corp,steadyvoice_noguard,guard_only,steadyvoice,sv_nowarm,sv_nopl,sv_noal,sv_nocons \
    > results/synthetic/log_s$s.txt 2>&1
done
echo ALL_DONE > results/synthetic/DONE
