#!/usr/bin/env bash
# Resumable full pipeline: (wait for stress) -> tune guard on dev seeds -> final runs on seeds 0-9. Relaunch after any container restart.
cd "$(dirname "$0")/.."
[ -f results/STRESS_DONE ] || scripts/run_guard_stress.sh
[ -f results/tune/best.json ] || python scripts/tune_guard.py > results/log_tune.txt 2>&1
python scripts/run_final.py --seeds 0,1,2,3,4,5,6,7,8,9 > results/log_final.txt 2>&1
echo done > results/PIPELINE_DONE
