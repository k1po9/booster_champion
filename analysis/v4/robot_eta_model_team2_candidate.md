# Schema v4 Robot ETA Training

Total ETA is supervised only by stable-target segments that actually arrived and started at least 0.25 m from the target. Validation holds out one whole match file.

- Travel arrivals: 277
- Fixed-interval suffix queries: 4726
- Near-target arrivals kept out of travel model: 924
- Selected: `balanced_ridge_0.1`
- Runtime median/p99: 8.0/26.5 us

| Model | n | median abs | p75 abs | p90 abs | signed median |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_ridge_0.1 | 4726 | 0.486s | 0.891s | 1.584s | -0.015s |
| balanced_ridge_1 | 4726 | 0.476s | 0.892s | 1.600s | -0.014s |
| ridge_1 | 4726 | 0.461s | 0.838s | 1.509s | 0.144s |
| ridge_0.1 | 4726 | 0.462s | 0.839s | 1.511s | 0.144s |
| balanced_ridge_10 | 4726 | 0.492s | 0.925s | 1.631s | -0.022s |
| ridge_10 | 4726 | 0.459s | 0.841s | 1.523s | 0.147s |
| knn_15 | 4726 | 0.390s | 0.845s | 1.852s | 0.026s |
| knn_9 | 4726 | 0.385s | 0.850s | 1.872s | 0.020s |
| knn_5 | 4726 | 0.391s | 0.874s | 1.942s | 0.011s |
| serial_kinematic | 4726 | 0.602s | 1.299s | 2.673s | 0.075s |
| log_ridge_1 | 4726 | 0.734s | 1.645s | 3.257s | 0.036s |
| log_ridge_0.1 | 4726 | 0.735s | 1.640s | 3.268s | 0.037s |
| log_ridge_10 | 4726 | 0.732s | 1.636s | 3.351s | 0.026s |
| parallel_kinematic | 4726 | 1.428s | 2.315s | 3.434s | -1.219s |
| distance_only | 4726 | 1.942s | 3.425s | 4.446s | -1.942s |

## Original Episode Starts (selected model)

n=277, median=0.688s, p75=1.371s, p90=2.439s, signed median=0.218s.

This result is not yet an interface approval. The small number of independent complete travel arrivals and the separate near-target alignment mode must be considered before strategy integration.
