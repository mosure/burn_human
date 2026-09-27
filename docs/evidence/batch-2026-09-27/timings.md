# Full-clip throughput

Warm medians, seconds per complete batch. Text encoding and loading excluded.

| Case | Actors × frames | Native serial / batch | Native gain | Browser serial / batch | Browser gain |
| --- | --- | --- | --- | --- | --- |
| single | 1 × 40 | 0.231 / 0.227 | 1.02× | 0.271 / 0.365 | 0.74× |
| partial | 2 × 44 | 0.922 / 0.568 | 1.62× | 0.828 / 0.434 | 1.91× |
| no_history | 2 × 80 | 0.882 / 0.486 | 1.81× | 0.906 / 0.456 | 1.99× |
| dense_curves | 4 × 120 | 2.926 / 0.945 | 3.09× | 2.505 / 0.866 | 2.89× |
| sparse_long | 2 × 240 | 2.908 / 1.822 | 1.60× | 2.725 / 1.698 | 1.60× |
| eight_actors | 8 × 40 | 1.840 / 0.300 | 6.13× | 1.688 / 0.268 | 6.29× |
| duplicate_seed | 2 × 40 | 0.440 / 0.216 | 2.04× | 0.391 / 0.198 | 1.98× |

The one-actor paths are the same implementation; their ratio reflects timing noise.
Native and browser profiles differ. See the parent report for hardware, methodology,
FSQ reproducibility limits and whole-run event-loop stalls. All three timing samples
are retained in the JSON reports.
