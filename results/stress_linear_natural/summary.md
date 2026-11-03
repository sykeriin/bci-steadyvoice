Seeds: 5 (0, 1, 2, 3, 4). mean ± std across seeds. LM-off phoneme error rate.

| method | mean PER | PER moderate-drift days | PER severe-drift days | worst-day PER | days >2x oracle | labels used | collapse events | reject/rollback days |
|---|---|---|---|---|---|---|---|---|
| none | 0.569 ± 0.157 | 0.049 ± 0.030 | 0.800 ± 0.099 | 0.911 ± 0.045 | 16.8 ± 1.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oracle | 0.012 ± 0.005 | 0.008 ± 0.004 | 0.014 ± 0.009 | 0.034 ± 0.023 | 0.0 ± 0.0 | 420.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| periodic7 | 0.125 ± 0.051 | 0.017 ± 0.007 | 0.188 ± 0.117 | 0.689 ± 0.228 | 9.8 ± 3.1 | 80.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| corp | 0.551 ± 0.319 | 0.008 ± 0.004 | 0.761 ± 0.378 | 0.800 ± 0.391 | 11.8 ± 6.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| steadyvoice_noguard | 0.222 ± 0.178 | 0.013 ± 0.008 | 0.365 ± 0.334 | 0.667 ± 0.337 | 7.4 ± 4.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| guard_only | 0.213 ± 0.176 | 0.049 ± 0.030 | 0.304 ± 0.287 | 0.489 ± 0.226 | 13.6 ± 1.6 | 44.0 ± 29.4 | 0.0 ± 0.0 | 18.8 ± 1.5 |
| steadyvoice | 0.143 ± 0.211 | 0.023 ± 0.019 | 0.221 ± 0.329 | 0.315 ± 0.342 | 5.0 ± 5.9 | 24.0 ± 23.3 | 0.0 ± 0.0 | 7.8 ± 6.8 |

Paired per-seed mean PER (rows = seeds):

| seed | none | oracle | periodic7 | corp | steadyvoice_noguard | guard_only | steadyvoice |
|---|---|---|---|---|---|---|---|
| 0 | 0.562 | 0.012 | 0.106 | 0.606 | 0.492 | 0.562 | 0.562 |
| 1 | 0.358 | 0.004 | 0.044 | 0.007 | 0.010 | 0.170 | 0.013 |
| 2 | 0.767 | 0.017 | 0.142 | 0.891 | 0.100 | 0.129 | 0.032 |
| 3 | 0.437 | 0.017 | 0.202 | 0.420 | 0.365 | 0.111 | 0.083 |
| 4 | 0.719 | 0.007 | 0.130 | 0.828 | 0.143 | 0.092 | 0.026 |
Paired difference in mean PER vs guarded SteadyVoice (positive = method worse). 95% bootstrap CI over seeds; with few seeds this is coarse.

| method | mean diff | 95% CI | seeds worse / total |
|---|---|---|---|
| none | +0.426 | [+0.208, +0.642] | 4/5 |
| oracle | -0.131 | [-0.346, -0.013] | 0/5 |
| periodic7 | -0.018 | [-0.243, +0.113] | 4/5 |
| corp | +0.407 | [+0.083, +0.732] | 4/5 |
| steadyvoice_noguard | +0.079 | [-0.015, +0.192] | 3/5 |
| guard_only | +0.070 | [+0.024, +0.119] | 4/5 |