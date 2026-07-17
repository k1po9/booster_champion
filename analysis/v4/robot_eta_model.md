# Schema v4 Robot ETA Training

Total ETA is supervised only by stable-target segments that actually arrived and started at least 0.25 m from the target. Validation holds out one whole match file.

- Travel arrivals: 88
- Fixed-interval suffix queries: 1834
- Near-target arrivals kept out of travel model: 621
- Selected: `balanced_ridge_0.1`
- Runtime median/p99: 9.0/36.8 us

| Model | n | median abs | p75 abs | p90 abs | signed median |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_ridge_0.1 | 1834 | 0.392s | 0.687s | 1.219s | -0.081s |
| balanced_ridge_1 | 1834 | 0.398s | 0.722s | 1.231s | -0.084s |
| ridge_10 | 1834 | 0.358s | 0.645s | 1.078s | 0.001s |
| ridge_1 | 1834 | 0.335s | 0.626s | 1.048s | 0.008s |
| ridge_0.1 | 1834 | 0.336s | 0.628s | 1.044s | 0.007s |
| balanced_ridge_10 | 1834 | 0.473s | 0.878s | 1.458s | -0.099s |
| knn_9 | 1834 | 0.318s | 0.636s | 1.203s | -0.001s |
| knn_5 | 1834 | 0.337s | 0.673s | 1.168s | 0.002s |
| knn_15 | 1834 | 0.338s | 0.661s | 1.314s | 0.001s |
| serial_kinematic | 1834 | 0.555s | 1.140s | 2.276s | 0.150s |
| log_ridge_0.1 | 1834 | 0.719s | 1.477s | 2.995s | 0.056s |
| log_ridge_1 | 1834 | 0.716s | 1.482s | 2.993s | 0.055s |
| log_ridge_10 | 1834 | 0.680s | 1.449s | 3.274s | 0.015s |
| parallel_kinematic | 1834 | 1.740s | 2.351s | 3.102s | -1.524s |
| distance_only | 1834 | 2.513s | 3.608s | 4.316s | -2.513s |

## Original Episode Starts (selected model)

n=88, median=0.540s, p75=0.912s, p90=1.729s, signed median=-0.108s.

This result is not yet an interface approval. The small number of independent complete travel arrivals and the separate near-target alignment mode must be considered before strategy integration.
