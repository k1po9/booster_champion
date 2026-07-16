# Ball Trajectory Challenger Training

All metrics are leave-one-match-file-out. Raw datasets are read-only.

- Clean free-roll tracks: 87
- Selected challenger: **piecewise_stable**
- Selected score: 1.061

## Held-Out Stop-Point Error

| Model | Observation | n | Median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_quadratic | 0.15s | 87 | 0.766m | 1.320m | 2.017m |
| current_quadratic | 0.25s | 87 | 0.645m | 1.252m | 1.837m |
| current_quadratic | 0.35s | 87 | 0.537m | 0.952m | 1.575m |
| current_quadratic | 0.5s | 87 | 0.485m | 0.789m | 1.398m |
| robust_quadratic | 0.15s | 87 | 0.763m | 1.320m | 2.017m |
| robust_quadratic | 0.25s | 87 | 0.647m | 1.247m | 1.831m |
| robust_quadratic | 0.35s | 87 | 0.524m | 0.958m | 1.582m |
| robust_quadratic | 0.5s | 87 | 0.487m | 0.798m | 1.443m |
| stable_quadratic | 0.15s | 87 | 0.776m | 1.323m | 2.018m |
| stable_quadratic | 0.25s | 87 | 0.651m | 1.251m | 1.838m |
| stable_quadratic | 0.35s | 87 | 0.524m | 0.977m | 1.582m |
| stable_quadratic | 0.5s | 87 | 0.494m | 0.805m | 1.432m |
| adaptive_quadratic | 0.15s | 87 | 0.779m | 1.342m | 2.055m |
| adaptive_quadratic | 0.25s | 87 | 0.678m | 1.339m | 1.959m |
| adaptive_quadratic | 0.35s | 87 | 0.647m | 1.048m | 1.807m |
| adaptive_quadratic | 0.5s | 87 | 0.598m | 0.947m | 1.882m |
| piecewise_stable | 0.15s | 87 | 0.696m | 1.341m | 2.029m |
| piecewise_stable | 0.25s | 87 | 0.618m | 1.210m | 1.820m |
| piecewise_stable | 0.35s | 87 | 0.527m | 0.900m | 1.542m |
| piecewise_stable | 0.5s | 87 | 0.489m | 0.758m | 1.458m |
| adaptive_piecewise_ensemble | 0.15s | 87 | 0.751m | 1.300m | 2.074m |
| adaptive_piecewise_ensemble | 0.25s | 87 | 0.640m | 1.310m | 1.942m |
| adaptive_piecewise_ensemble | 0.35s | 87 | 0.562m | 0.991m | 1.664m |
| adaptive_piecewise_ensemble | 0.5s | 87 | 0.569m | 0.801m | 1.601m |

## Held-Out Future-Position Error

The label `0.25+0.50` means 0.25s of observation followed by a 0.50s prediction.

| Model | Observation+Future | n | Median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_quadratic | 0.15+0.1s | 87 | 0.038m | 0.057m | 0.091m |
| current_quadratic | 0.15+0.25s | 87 | 0.060m | 0.098m | 0.191m |
| current_quadratic | 0.15+0.5s | 87 | 0.100m | 0.174m | 0.471m |
| current_quadratic | 0.25+0.1s | 87 | 0.033m | 0.051m | 0.078m |
| current_quadratic | 0.25+0.25s | 87 | 0.049m | 0.084m | 0.205m |
| current_quadratic | 0.25+0.5s | 87 | 0.078m | 0.173m | 0.383m |
| current_quadratic | 0.35+0.1s | 87 | 0.023m | 0.044m | 0.062m |
| current_quadratic | 0.35+0.25s | 87 | 0.036m | 0.072m | 0.119m |
| current_quadratic | 0.35+0.5s | 87 | 0.063m | 0.123m | 0.207m |
| current_quadratic | 0.5+0.1s | 87 | 0.029m | 0.047m | 0.076m |
| current_quadratic | 0.5+0.25s | 87 | 0.041m | 0.071m | 0.114m |
| current_quadratic | 0.5+0.5s | 87 | 0.054m | 0.108m | 0.193m |
| robust_quadratic | 0.15+0.1s | 87 | 0.038m | 0.057m | 0.090m |
| robust_quadratic | 0.15+0.25s | 87 | 0.058m | 0.098m | 0.191m |
| robust_quadratic | 0.15+0.5s | 87 | 0.100m | 0.174m | 0.460m |
| robust_quadratic | 0.25+0.1s | 87 | 0.033m | 0.051m | 0.073m |
| robust_quadratic | 0.25+0.25s | 87 | 0.049m | 0.084m | 0.201m |
| robust_quadratic | 0.25+0.5s | 87 | 0.078m | 0.173m | 0.380m |
| robust_quadratic | 0.35+0.1s | 87 | 0.023m | 0.044m | 0.062m |
| robust_quadratic | 0.35+0.25s | 87 | 0.036m | 0.066m | 0.119m |
| robust_quadratic | 0.35+0.5s | 87 | 0.063m | 0.109m | 0.209m |
| robust_quadratic | 0.5+0.1s | 87 | 0.029m | 0.047m | 0.077m |
| robust_quadratic | 0.5+0.25s | 87 | 0.041m | 0.071m | 0.116m |
| robust_quadratic | 0.5+0.5s | 87 | 0.053m | 0.107m | 0.187m |
| stable_quadratic | 0.15+0.1s | 87 | 0.038m | 0.057m | 0.090m |
| stable_quadratic | 0.15+0.25s | 87 | 0.058m | 0.098m | 0.191m |
| stable_quadratic | 0.15+0.5s | 87 | 0.100m | 0.174m | 0.460m |
| stable_quadratic | 0.25+0.1s | 87 | 0.033m | 0.051m | 0.070m |
| stable_quadratic | 0.25+0.25s | 87 | 0.049m | 0.083m | 0.201m |
| stable_quadratic | 0.25+0.5s | 87 | 0.080m | 0.167m | 0.380m |
| stable_quadratic | 0.35+0.1s | 87 | 0.022m | 0.044m | 0.065m |
| stable_quadratic | 0.35+0.25s | 87 | 0.036m | 0.065m | 0.119m |
| stable_quadratic | 0.35+0.5s | 87 | 0.064m | 0.109m | 0.206m |
| stable_quadratic | 0.5+0.1s | 87 | 0.029m | 0.046m | 0.077m |
| stable_quadratic | 0.5+0.25s | 87 | 0.042m | 0.069m | 0.115m |
| stable_quadratic | 0.5+0.5s | 87 | 0.055m | 0.109m | 0.179m |
| adaptive_quadratic | 0.15+0.1s | 87 | 0.038m | 0.057m | 0.090m |
| adaptive_quadratic | 0.15+0.25s | 87 | 0.059m | 0.099m | 0.191m |
| adaptive_quadratic | 0.15+0.5s | 87 | 0.100m | 0.176m | 0.460m |
| adaptive_quadratic | 0.25+0.1s | 87 | 0.033m | 0.051m | 0.071m |
| adaptive_quadratic | 0.25+0.25s | 87 | 0.052m | 0.086m | 0.201m |
| adaptive_quadratic | 0.25+0.5s | 87 | 0.087m | 0.164m | 0.386m |
| adaptive_quadratic | 0.35+0.1s | 87 | 0.021m | 0.044m | 0.066m |
| adaptive_quadratic | 0.35+0.25s | 87 | 0.036m | 0.069m | 0.123m |
| adaptive_quadratic | 0.35+0.5s | 87 | 0.059m | 0.110m | 0.214m |
| adaptive_quadratic | 0.5+0.1s | 87 | 0.030m | 0.046m | 0.078m |
| adaptive_quadratic | 0.5+0.25s | 87 | 0.042m | 0.073m | 0.118m |
| adaptive_quadratic | 0.5+0.5s | 87 | 0.066m | 0.124m | 0.203m |
| piecewise_stable | 0.15+0.1s | 87 | 0.038m | 0.057m | 0.090m |
| piecewise_stable | 0.15+0.25s | 87 | 0.058m | 0.097m | 0.191m |
| piecewise_stable | 0.15+0.5s | 87 | 0.093m | 0.165m | 0.457m |
| piecewise_stable | 0.25+0.1s | 87 | 0.033m | 0.051m | 0.070m |
| piecewise_stable | 0.25+0.25s | 87 | 0.048m | 0.082m | 0.200m |
| piecewise_stable | 0.25+0.5s | 87 | 0.073m | 0.160m | 0.375m |
| piecewise_stable | 0.35+0.1s | 87 | 0.021m | 0.044m | 0.065m |
| piecewise_stable | 0.35+0.25s | 87 | 0.036m | 0.063m | 0.117m |
| piecewise_stable | 0.35+0.5s | 87 | 0.054m | 0.104m | 0.211m |
| piecewise_stable | 0.5+0.1s | 87 | 0.029m | 0.046m | 0.076m |
| piecewise_stable | 0.5+0.25s | 87 | 0.040m | 0.067m | 0.111m |
| piecewise_stable | 0.5+0.5s | 87 | 0.057m | 0.107m | 0.172m |
| adaptive_piecewise_ensemble | 0.15+0.1s | 87 | 0.038m | 0.057m | 0.090m |
| adaptive_piecewise_ensemble | 0.15+0.25s | 87 | 0.058m | 0.098m | 0.191m |
| adaptive_piecewise_ensemble | 0.15+0.5s | 87 | 0.100m | 0.170m | 0.459m |
| adaptive_piecewise_ensemble | 0.25+0.1s | 87 | 0.033m | 0.051m | 0.070m |
| adaptive_piecewise_ensemble | 0.25+0.25s | 87 | 0.049m | 0.084m | 0.201m |
| adaptive_piecewise_ensemble | 0.25+0.5s | 87 | 0.080m | 0.161m | 0.381m |
| adaptive_piecewise_ensemble | 0.35+0.1s | 87 | 0.021m | 0.044m | 0.065m |
| adaptive_piecewise_ensemble | 0.35+0.25s | 87 | 0.037m | 0.066m | 0.120m |
| adaptive_piecewise_ensemble | 0.35+0.5s | 87 | 0.054m | 0.102m | 0.212m |
| adaptive_piecewise_ensemble | 0.5+0.1s | 87 | 0.029m | 0.046m | 0.077m |
| adaptive_piecewise_ensemble | 0.5+0.25s | 87 | 0.041m | 0.070m | 0.115m |
| adaptive_piecewise_ensemble | 0.5+0.5s | 87 | 0.060m | 0.117m | 0.189m |

## Inference Microbenchmark

| Model | Median | p99 | Max |
| --- | ---: | ---: | ---: |
| current_quadratic | 114.0us | 321.6us | 480.8us |
| robust_quadratic | 349.4us | 899.2us | 1323.6us |
| stable_quadratic | 295.7us | 956.1us | 1383.7us |
| adaptive_quadratic | 288.2us | 838.7us | 1631.9us |
| piecewise_stable | 279.6us | 788.1us | 1281.7us |
| adaptive_piecewise_ensemble | 302.2us | 926.1us | 1702.9us |

## Selection

The score combines 0.35s stop median/p75 with two 0.25s-ahead path medians.
Selection is evidence for the next experiment, not authorization for strategy takeover.
