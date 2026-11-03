#!/usr/bin/env bash
# Resumable. (1) finish dev-seed tuning, (2) final runs seeds 0-9, (3) fresh-seed (5-9) evaluation of the reliability
# fallback in the natural-only condition, with fallback ON vs OFF (control). Relaunch after any container restart.
cd "$(dirname "$0")/.."
M=none,oracle,periodic7,guard_only,steadyvoice
[ -f results/tune/best.json ] || python scripts/tune_guard.py > results/log_tune.txt 2>&1
python scripts/run_final.py --seeds 0,1,2,3,4,5,6,7,8,9 > results/log_final.txt 2>&1
python scripts/run_final.py --seeds 5,6,7,8,9 --natural-only --methods $M --gaps "" --probe "" --out results/natural_fallback_on  > results/log_nat_on.txt 2>&1
python scripts/run_final.py --seeds 5,6,7,8,9 --natural-only --methods $M --gaps "" --probe "" --min-rho -1 --out results/natural_fallback_off > results/log_nat_off.txt 2>&1
echo done > results/PIPELINE_DONE
