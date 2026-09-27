# Complete-clip throughput

Warm medians in seconds, including decoding and host-visible clips. Text encoding
and weight loading are excluded. Three alternating serial/batch samples per case
are retained in the reports. Gains compare batching against the same candidate's
serial path.

| Case | Actors × frames | Native serial / batch | Gain | Browser serial / batch | Gain |
| --- | --- | --- | --- | --- | --- |
| single | 1 × 40 | 0.217 / 0.222 | 0.98× | 0.213 / 0.203 | 1.05× |
| partial | 2 × 44 | 0.901 / 0.465 | 1.94× | 0.719 / 0.414 | 1.74× |
| no_history | 2 × 80 | 0.878 / 0.519 | 1.69× | 0.722 / 0.417 | 1.73× |
| dense_curves | 4 × 120 | 2.883 / 1.235 | 2.33× | 2.537 / 1.045 | 2.43× |
| sparse_long | 2 × 240 | 3.066 / 2.079 | 1.47× | 2.944 / 2.036 | 1.45× |
| eight_actors | 8 × 40 | 1.823 / 0.328 | 5.56× | 1.378 / 0.270 | 5.11× |
| duplicate_seed | 2 × 40 | 0.437 / 0.221 | 1.98× | 0.315 / 0.199 | 1.58× |
| actors_3 | 3 × 40 | 0.677 / 0.262 | 2.59× | 0.539 / 0.201 | 2.68× |
| actors_5 | 5 × 40 | 1.177 / 0.302 | 3.89× | 0.951 / 0.252 | 3.77× |
| actors_6 | 6 × 40 | 1.392 / 0.333 | 4.17× | 1.034 / 0.269 | 3.84× |
| actors_7 | 7 × 40 | 1.602 / 0.345 | 4.65× | 1.204 / 0.267 | 4.51× |

Fixed kernels have a throughput cost in some native long-batch cases. The
previous qualification measured 0.945 s for the dense-curve batch; this run
measures 1.235 s. The partial-window batch improves from 0.568 s to 0.465 s.
These are separate runs on a shared machine, not a controlled paired comparison.
The reproducibility gates are now strict in every case. See the parent report
for adapter, build profiles and measurement boundaries.
