Seeds: 5 (5, 6, 7, 8, 9). mean ± std across seeds. LM-off phoneme error rate.

| method | mean PER | PER moderate-drift days | PER severe-drift days | worst-day PER | days >2x oracle | labels used | collapse events | reject/rollback days |
|---|---|---|---|---|---|---|---|---|
| none | 0.483 ± 0.105 | 0.063 ± 0.022 | 0.790 ± 0.159 | 0.892 ± 0.111 | 15.8 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oracle | 0.011 ± 0.002 | 0.007 ± 0.003 | 0.014 ± 0.003 | 0.041 ± 0.016 | 0.0 ± 0.0 | 420.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| periodic7 | 0.124 ± 0.072 | 0.018 ± 0.003 | 0.199 ± 0.116 | 0.738 ± 0.252 | 8.8 ± 1.9 | 76.0 ± 8.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| guard_only | 0.079 ± 0.010 | 0.054 ± 0.021 | 0.096 ± 0.023 | 0.244 ± 0.048 | 11.4 ± 1.0 | 56.0 ± 15.0 | 0.0 ± 0.0 | 18.2 ± 0.7 |
| steadyvoice | 0.021 ± 0.005 | 0.012 ± 0.003 | 0.028 ± 0.007 | 0.070 ± 0.024 | 1.2 ± 1.0 | 16.0 ± 8.0 | 0.0 ± 0.0 | 2.0 ± 1.1 |

Paired per-seed mean PER (rows = seeds):

| seed | none | oracle | periodic7 | guard_only | steadyvoice |
|---|---|---|---|---|---|
| 5 | 0.438 | 0.015 | 0.079 | 0.071 | 0.028 |
| 6 | 0.527 | 0.011 | 0.176 | 0.068 | 0.026 |
| 7 | 0.311 | 0.010 | 0.050 | 0.078 | 0.016 |
| 8 | 0.516 | 0.008 | 0.077 | 0.096 | 0.018 |
| 9 | 0.625 | 0.011 | 0.241 | 0.083 | 0.020 |
Paired difference in mean PER vs guarded SteadyVoice (positive = method worse). 95% bootstrap CI over seeds; with few seeds this is coarse.

| method | mean diff | 95% CI | seeds worse / total |
|---|---|---|---|
| none | +0.462 | [+0.376, +0.545] | 5/5 |
| oracle | -0.010 | [-0.013, -0.008] | 0/5 |
| periodic7 | +0.103 | [+0.046, +0.173] | 5/5 |
| guard_only | +0.058 | [+0.047, +0.069] | 5/5 |