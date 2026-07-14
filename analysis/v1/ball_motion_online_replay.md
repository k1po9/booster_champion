# Online Ball Predictor Replay

This report replays the 87 clean free-roll tracks through the actual `src/tactics/ball_prediction.py` sliding-window implementation and its default quadratic resistance model. It complements the model-selection cross-validation by checking the code that will be called online.

| Observation span | n | Median stop error | p75 stop error |
| --- | ---: | ---: | ---: |
| 0.15 s | 87 | 0.771 m | 1.296 m |
| 0.25 s | 87 | 0.648 m | 1.243 m |
| 0.35 s | 87 | 0.537 m | 1.063 m |
| 0.50 s | 87 | 0.470 m | 0.902 m |

The online predictor preserves the 0.35 s median result of the cross-validation experiment. Its p75 tail is slightly more conservative because it uses the production sliding window and discontinuity resets. `uncertainty_radius_m` is calibrated to this replay envelope. The capability remains advisory and is not yet wired as the sole chaser target.
