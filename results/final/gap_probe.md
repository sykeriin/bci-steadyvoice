Day-gap jump: adapt ONCE directly on day k+g-1 (intermediate days unseen). PER mean ± std over 10 seeds (labels used in brackets).

| method | g=1 | g=5 | g=10 | g=20 |
|---|---|---|---|---|
| none | 0.006 ± 0.006 [0] | 0.175 ± 0.288 [0] | 0.541 ± 0.303 [0] | 0.882 ± 0.083 [0] |
| oracle | 0.005 ± 0.005 [20] | 0.012 ± 0.011 [20] | 0.023 ± 0.014 [20] | 0.056 ± 0.042 [20] |
| periodic7 | 0.006 ± 0.006 [0] | 0.175 ± 0.288 [0] | 0.541 ± 0.303 [0] | 0.882 ± 0.083 [0] |
| corp | 0.005 ± 0.005 [0] | 0.193 ± 0.364 [0] | 0.492 ± 0.452 [0] | 0.964 ± 0.035 [0] |
| steadyvoice_noguard | 0.008 ± 0.007 [0] | 0.138 ± 0.247 [0] | 0.423 ± 0.378 [0] | 0.841 ± 0.137 [0] |
| guard_only | 0.006 ± 0.006 [0] | 0.031 ± 0.013 [4] | 0.041 ± 0.056 [18] | 0.056 ± 0.042 [20] |
| steadyvoice | 0.006 ± 0.006 [0] | 0.021 ± 0.013 [4] | 0.076 ± 0.040 [10] | 0.178 ± 0.157 [20] |

Linear probe (analysis only, uses labels): fresh linear CTC head refit on the day's labelled trials, on the adapted latents.
probe << frozen-head PER => information preserved, readout misaligned; probe high => information destroyed.

| method | frozen-head PER | probe PER | days |
|---|---|---|---|
| none | 0.526 | 0.179 | 210 |
| steadyvoice_noguard | 0.264 | 0.155 | 210 |
| steadyvoice | 0.024 | 0.026 | 210 |