#!/usr/bin/env bash
# Guard stress: {linear,nonlinear} drift x {full,natural-only} predictor. (linear+full = main run.)
cd "$(dirname "$0")/.."
M=none,oracle,periodic7,corp,steadyvoice_noguard,guard_only,steadyvoice
for s in 0 1 2 3 4; do
  python scripts/run_synthetic.py --seed $s --days 26 --methods $M --drift nonlinear --out results/stress_nonlinear_full > results/log_nl_full_$s.txt 2>&1
  python scripts/run_synthetic.py --seed $s --days 26 --methods $M --natural-only --out results/stress_linear_natural > results/log_lin_nat_$s.txt 2>&1
  python scripts/run_synthetic.py --seed $s --days 26 --methods $M --drift nonlinear --natural-only --out results/stress_nonlinear_natural > results/log_nl_nat_$s.txt 2>&1
done
echo done > results/STRESS_DONE
