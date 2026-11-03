Seeds: 5 (0, 1, 2, 3, 4). mean ± std across seeds. LM-off phoneme error rate.

| method | mean PER | PER moderate-drift days | PER severe-drift days | worst-day PER | days >2x oracle | labels used | collapse events | reject/rollback days |
|---|---|---|---|---|---|---|---|---|
| none | 0.505 ± 0.089 | 0.078 ± 0.031 | 0.766 ± 0.126 | 0.902 ± 0.050 | 17.2 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oracle | 0.018 ± 0.007 | 0.008 ± 0.002 | 0.025 ± 0.013 | 0.096 ± 0.071 | 0.0 ± 0.0 | 420.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| periodic7 | 0.172 ± 0.109 | 0.034 ± 0.019 | 0.270 ± 0.220 | 0.532 ± 0.293 | 12.5 ± 1.8 | 84.0 ± 8.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| corp | 0.264 ± 0.289 | 0.008 ± 0.003 | 0.422 ± 0.458 | 0.605 ± 0.457 | 7.0 ± 6.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| steadyvoice_noguard | 0.239 ± 0.254 | 0.012 ± 0.003 | 0.380 ± 0.405 | 0.562 ± 0.424 | 8.8 ± 5.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| guard_only | 0.102 ± 0.012 | 0.061 ± 0.019 | 0.125 ± 0.019 | 0.308 ± 0.048 | 13.0 ± 1.2 | 84.0 ± 15.0 | 0.0 ± 0.0 | 16.8 ± 0.7 |
| steadyvoice | 0.030 ± 0.005 | 0.015 ± 0.003 | 0.040 ± 0.011 | 0.085 ± 0.024 | 3.0 ± 2.9 | 20.0 ± 17.9 | 0.0 ± 0.0 | 3.6 ± 1.2 |

Paired per-seed mean PER (rows = seeds):

| seed | none | oracle | periodic7 | corp | steadyvoice_noguard | guard_only | steadyvoice |
|---|---|---|---|---|---|---|---|
| 0 | 0.448 | 0.015 | 0.072 | 0.058 | 0.055 | 0.106 | 0.025 |
| 1 | 0.677 | 0.018 | 0.199 | 0.704 | 0.621 | 0.121 | 0.034 |
| 2 | 0.451 | 0.014 | 0.090 | 0.018 | 0.020 | 0.102 | 0.025 |
| 3 | 0.442 | 0.012 | 0.128 | 0.024 | 0.032 | 0.085 | 0.033 |
| 4 | 0.507 | 0.030 | 0.371 | 0.517 | 0.468 | 0.095 | 0.036 |
Paired difference in mean PER vs guarded SteadyVoice (positive = method worse). 95% bootstrap CI over seeds; with few seeds this is coarse.

| method | mean diff | 95% CI | seeds worse / total |
|---|---|---|---|
| none | +0.474 | [+0.418, +0.557] | 5/5 |
| oracle | -0.013 | [-0.017, -0.008] | 0/5 |
| periodic7 | +0.142 | [+0.064, +0.247] | 5/5 |
| corp | +0.234 | [+0.000, +0.497] | 3/5 |
| steadyvoice_noguard | +0.209 | [+0.004, +0.439] | 3/5 |
| guard_only | +0.071 | [+0.059, +0.083] | 5/5 |