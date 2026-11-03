#!/usr/bin/env bash
# Guard stress: {linear,nonlinear} drift x {full,natural-only} predictor. RESUMABLE: skips finished runs
# (the build container restarts occasionally and kills background jobs; just relaunch).
cd "$(dirname "$0")/.."
M=none,oracle,periodic7,corp,steadyvoice_noguard,guard_only,steadyvoice
run() { # name drift extra seed
  out=results/stress_$1; f=$out/results_k5_s$4.json
  [ -f "$f" ] && return
  python scripts/run_synthetic.py --seed $4 --days 26 --methods $M --drift $2 $3 --out $out > results/log_$1_$4.txt 2>&1
}
for s in 0 1 2 3 4; do
  run nonlinear_full nonlinear "" $s
  run linear_natural linear --natural-only $s
  run nonlinear_natural nonlinear --natural-only $s
done
echo done > results/STRESS_DONE
