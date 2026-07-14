# Robot ETA Cross-Validation

This experiment validates locomotion components from recorded poses and executed commands. The logs do not contain the strategy internal target point, so this is not yet arbitrary-target ETA validation.

## Dataset

- Translation episodes: 254
- From-rest translation episodes: 6
- Turn episodes: 818
- Cross-validation unit: one complete match JSONL file.
- Cruise models use all straight sustained episodes; the dynamics model uses only episodes detected as starting from rest.
- Translation validation uses actual accumulated path distance.

## Held-Out Translation ETA Absolute Error

| Model | Distance | n | Median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.25m | 254 | 0.078s | 0.131s | 0.234s |
| baseline | 0.5m | 182 | 0.118s | 0.197s | 0.286s |
| baseline | 1.0m | 97 | 0.195s | 0.267s | 0.352s |
| empirical_cruise | 0.25m | 254 | 0.042s | 0.094s | 0.197s |
| empirical_cruise | 0.5m | 182 | 0.047s | 0.125s | 0.214s |
| empirical_cruise | 1.0m | 97 | 0.056s | 0.129s | 0.206s |
| dynamics | 0.25m | 6 | 0.331s | 0.365s | 0.455s |
| dynamics | 0.5m | 4 | 0.267s | 0.346s | 0.461s |
| dynamics | 1.0m | 1 | 0.168s | 0.168s | 0.168s |

## Held-Out Turn ETA Absolute Error

| Model | Rotation | n | Median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| command_rate | 0.25rad | 818 | 0.232s | 0.317s | 0.450s |
| command_rate | 0.5rad | 715 | 0.247s | 0.352s | 0.477s |
| command_rate | 1.0rad | 521 | 0.313s | 0.416s | 0.545s |
| empirical_rate | 0.25rad | 818 | 0.199s | 0.281s | 0.415s |
| empirical_rate | 0.5rad | 715 | 0.179s | 0.294s | 0.419s |
| empirical_rate | 1.0rad | 521 | 0.178s | 0.292s | 0.410s |

## Fold Calibration

### Held out: `match_dataset_1.jsonl`

- gain=0.898, reaction=0.354s, acceleration=1.429m/s², yaw_gain=0.882; validation translation/rest/turn=49/0/140

### Held out: `match_dataset_2.jsonl`

- gain=0.892, reaction=0.203s, acceleration=1.768m/s², yaw_gain=0.876; validation translation/rest/turn=53/1/152

### Held out: `match_dataset_3.jsonl`

- gain=0.895, reaction=0.500s, acceleration=1.429m/s², yaw_gain=0.879; validation translation/rest/turn=71/2/166

### Held out: `match_dataset_4.jsonl`

- gain=0.894, reaction=0.203s, acceleration=1.429m/s², yaw_gain=0.878; validation translation/rest/turn=43/1/182

### Held out: `match_dataset_5.jsonl`

- gain=0.895, reaction=0.304s, acceleration=1.344m/s², yaw_gain=0.879; validation translation/rest/turn=38/2/178

## Full-Data Calibration

- Translation command gain: 0.8945
- Start reaction delay: 0.3544s
- Acceleration: 1.4294m/s²
- Yaw command gain: 0.8789

## Scope Judgment

- This calibration can support a first straight-line ETA prior and heading-turn cost.
- A deployable target ETA still needs per-tick target coordinates, adjusted avoidance waypoint/path length, current velocity estimation, and arrival/slowdown labels.
- Do not yet use this report alone to decide a hard ball-interception point.
