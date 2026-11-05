Seeds: 5 (5, 6, 7, 8, 9). mean ± std across seeds. LM-off phoneme error rate.

| method | mean PER | PER moderate-drift days | PER severe-drift days | worst-day PER | days >2x oracle | labels used | collapse events | reject/rollback days |
|---|---|---|---|---|---|---|---|---|
| none | 0.483 ± 0.105 | 0.063 ± 0.022 | 0.790 ± 0.159 | 0.892 ± 0.111 | 15.8 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oracle | 0.011 ± 0.002 | 0.007 ± 0.003 | 0.014 ± 0.003 | 0.041 ± 0.016 | 0.0 ± 0.0 | 420.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| periodic7 | 0.124 ± 0.072 | 0.018 ± 0.003 | 0.199 ± 0.116 | 0.738 ± 0.252 | 8.8 ± 1.9 | 76.0 ± 8.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| guard_only | 0.081 ± 0.009 | 0.041 ± 0.017 | 0.109 ± 0.016 | 0.376 ± 0.259 | 11.0 ± 1.7 | 60.0 ± 17.9 | 0.0 ± 0.0 | 14.6 ± 7.3 |
| steadyvoice | 0.031 ± 0.024 | 0.013 ± 0.003 | 0.045 ± 0.039 | 0.225 ± 0.330 | 2.2 ± 2.9 | 28.0 ± 27.1 | 0.0 ± 0.0 | 1.6 ± 1.4 |

Paired per-seed mean PER (rows = seeds):

| seed | none | oracle | periodic7 | guard_only | steadyvoice |
|---|---|---|---|---|---|
| 5 | 0.438 | 0.015 | 0.079 | 0.079 | 0.079 |
| 6 | 0.527 | 0.011 | 0.176 | 0.068 | 0.026 |
| 7 | 0.311 | 0.010 | 0.050 | 0.078 | 0.016 |
| 8 | 0.516 | 0.008 | 0.077 | 0.096 | 0.018 |
| 9 | 0.625 | 0.011 | 0.241 | 0.083 | 0.020 |
Paired difference in mean PER vs guarded SteadyVoice (positive = method worse). 95% bootstrap CI over seeds; with few seeds this is coarse.

| method | mean diff | 95% CI | seeds worse / total |
|---|---|---|---|
| none | +0.452 | [+0.357, +0.542] | 5/5 |
| oracle | -0.021 | [-0.042, -0.008] | 0/5 |
| periodic7 | +0.093 | [+0.026, +0.170] | 4/5 |
| guard_only | +0.049 | [+0.024, +0.069] | 4/5 |