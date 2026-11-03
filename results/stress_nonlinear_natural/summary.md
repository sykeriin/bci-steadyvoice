Seeds: 5 (0, 1, 2, 3, 4). mean ± std across seeds. LM-off phoneme error rate.

| method | mean PER | PER moderate-drift days | PER severe-drift days | worst-day PER | days >2x oracle | labels used | collapse events | reject/rollback days |
|---|---|---|---|---|---|---|---|---|
| none | 0.505 ± 0.089 | 0.078 ± 0.031 | 0.766 ± 0.126 | 0.902 ± 0.050 | 17.4 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oracle | 0.018 ± 0.007 | 0.008 ± 0.002 | 0.025 ± 0.013 | 0.096 ± 0.071 | 0.0 ± 0.0 | 420.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| periodic7 | 0.172 ± 0.109 | 0.034 ± 0.019 | 0.270 ± 0.220 | 0.532 ± 0.293 | 12.2 ± 1.7 | 84.0 ± 8.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| corp | 0.264 ± 0.289 | 0.008 ± 0.003 | 0.422 ± 0.458 | 0.605 ± 0.457 | 5.8 ± 6.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| steadyvoice_noguard | 0.239 ± 0.254 | 0.012 ± 0.003 | 0.380 ± 0.405 | 0.562 ± 0.424 | 7.4 ± 5.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| guard_only | 0.152 ± 0.026 | 0.078 ± 0.031 | 0.199 ± 0.050 | 0.462 ± 0.099 | 14.6 ± 1.0 | 60.0 ± 17.9 | 0.0 ± 0.0 | 18.0 ± 0.9 |
| steadyvoice | 0.031 ± 0.005 | 0.018 ± 0.004 | 0.039 ± 0.011 | 0.086 ± 0.016 | 2.8 ± 2.8 | 20.0 ± 17.9 | 0.0 ± 0.0 | 4.0 ± 0.9 |

Paired per-seed mean PER (rows = seeds):

| seed | none | oracle | periodic7 | corp | steadyvoice_noguard | guard_only | steadyvoice |
|---|---|---|---|---|---|---|---|
| 0 | 0.448 | 0.015 | 0.072 | 0.058 | 0.055 | 0.134 | 0.025 |
| 1 | 0.677 | 0.018 | 0.199 | 0.704 | 0.621 | 0.135 | 0.034 |
| 2 | 0.451 | 0.014 | 0.090 | 0.018 | 0.020 | 0.142 | 0.026 |
| 3 | 0.442 | 0.012 | 0.128 | 0.024 | 0.032 | 0.203 | 0.033 |
| 4 | 0.507 | 0.030 | 0.371 | 0.517 | 0.468 | 0.147 | 0.038 |
Paired difference in mean PER vs guarded SteadyVoice (positive = method worse). 95% bootstrap CI over seeds; with few seeds this is coarse.

| method | mean diff | 95% CI | seeds worse / total |
|---|---|---|---|
| none | +0.474 | [+0.418, +0.556] | 5/5 |
| oracle | -0.013 | [-0.018, -0.009] | 0/5 |
| periodic7 | +0.141 | [+0.064, +0.246] | 5/5 |
| corp | +0.233 | [-0.000, +0.496] | 3/5 |
| steadyvoice_noguard | +0.208 | [+0.004, +0.438] | 3/5 |
| guard_only | +0.121 | [+0.106, +0.145] | 5/5 |