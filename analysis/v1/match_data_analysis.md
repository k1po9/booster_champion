# Match Dataset Analysis

This report is generated from immutable raw JSONL logs under `dataset/`.

## Dataset Quality

### match_dataset_1.jsonl

- Records: {'metadata': 1, 'frame': 16997, 'kick': 39}
- Duration: 1729.51s
- Frame gap median/p95/max: 0.101s / 0.104s / 1.151s
- Ball visible ratio: 1.000
- Pose missing ratio: 0.00000
- Top game states: [('FINISHED/NONE', 10232), ('PLAYING/NONE', 3250), ('READY/NONE', 2217), ('PLAYING/GOAL_KICK', 473), ('SET/NONE', 361), ('PLAYING/THROW_IN', 239), ('INITIAL/NONE', 123), ('PLAYING/CORNER_KICK', 99)]
- Top command reasons: [('non playing state', 32148), ('supporter hold', 1949), ('ready keeper: arrived', 1685), ('center kick to target', 1406), ('ready side: arrived', 1309)]

### match_dataset_2.jsonl

- Records: {'metadata': 1, 'frame': 11386, 'kick': 35}
- Duration: 1158.09s
- Frame gap median/p95/max: 0.101s / 0.104s / 1.109s
- Ball visible ratio: 1.000
- Pose missing ratio: 0.00026
- Top game states: [('FINISHED/NONE', 5103), ('PLAYING/NONE', 3778), ('READY/NONE', 1850), ('PLAYING/THROW_IN', 324), ('SET/NONE', 299), ('INITIAL/NONE', 25), ('unknown/none', 7)]
- Top command reasons: [('non playing state', 16281), ('supporter hold', 2327), ('center kick to target', 1811), ('ready keeper: arrived', 1218), ('supporter hold turn', 1217)]

### match_dataset_3.jsonl

- Records: {'metadata': 1, 'frame': 13279, 'kick': 56}
- Duration: 1349.55s
- Frame gap median/p95/max: 0.101s / 0.104s / 0.418s
- Ball visible ratio: 1.000
- Pose missing ratio: 0.00008
- Top game states: [('FINISHED/NONE', 6769), ('PLAYING/NONE', 3789), ('READY/NONE', 1517), ('PLAYING/THROW_IN', 497), ('SET/NONE', 290), ('PLAYING/CORNER_KICK', 229), ('PLAYING/GOAL_KICK', 144), ('INITIAL/NONE', 43)]
- Top command reasons: [('non playing state', 21306), ('supporter hold', 2020), ('center kick to target', 1610), ('goalkeeper guard', 1125), ('supporter hold turn', 1095)]

### match_dataset_4.jsonl

- Records: {'metadata': 1, 'frame': 7803, 'kick': 46}
- Duration: 794.09s
- Frame gap median/p95/max: 0.101s / 0.104s / 1.182s
- Ball visible ratio: 1.000
- Pose missing ratio: 0.00000
- Top game states: [('PLAYING/NONE', 4139), ('READY/NONE', 1588), ('FINISHED/NONE', 1547), ('PLAYING/THROW_IN', 273), ('SET/NONE', 240), ('INITIAL/NONE', 15), ('unknown/none', 1)]
- Top command reasons: [('non playing state', 5406), ('supporter hold', 1822), ('center kick to target', 1676), ('center kick', 1260), ('supporter hold turn', 1234)]

### match_dataset_5.jsonl

- Records: {'metadata': 1, 'frame': 7215, 'kick': 38}
- Duration: 732.53s
- Frame gap median/p95/max: 0.101s / 0.104s / 0.297s
- Ball visible ratio: 1.000
- Pose missing ratio: 0.00002
- Top game states: [('PLAYING/NONE', 3931), ('READY/NONE', 1548), ('FINISHED/NONE', 813), ('PLAYING/THROW_IN', 429), ('SET/NONE', 244), ('PLAYING/CORNER_KICK', 229), ('INITIAL/NONE', 20), ('unknown/none', 1)]
- Top command reasons: [('non playing state', 3231), ('supporter hold', 2503), ('center kick to target', 1931), ('ready keeper: arrived', 1133), ('supporter hold turn', 1130)]

## Ball Model

- Kicks analyzed: 206
- Global median deceleration: 3.599 m/s^2
- Stop prediction error by observation horizon:
  - 0.5s: n=154, median=0.911m, p75=1.982m, p90=3.244m
  - 1.0s: n=156, median=0.645m, p75=1.412m, p90=2.457m
  - 1.5s: n=159, median=0.479m, p75=1.273m, p90=2.342m
  - 2.0s: n=161, median=0.285m, p75=1.043m, p90=2.121m

## Kick Profile

- Estimated initial speed median: 2.443 m/s
- Peak speed median: 2.537 m/s
- Straight distance median: 2.343 m
- Path distance median: 2.689 m
- Kick end reasons: {'ball_settled': 123, 'timeout': 25, 'next_kick': 58}

## Kick Power To Speed

- power=1.5: n=206, initial speed median/p25/p75=2.443/1.673/3.052 m/s, straight distance median=2.343 m
- Power curve status: not enough distinct kick_power levels yet; collect multi-power calibration kicks before using this to choose arbitrary kick force.

## Robot Motion Profile

- match_dataset_1.jsonl: speed median/p75/p95=0.000/0.079/0.716 m/s, yaw median/p75/p95=0.000/0.047/0.881 rad/s
- match_dataset_2.jsonl: speed median/p75/p95=0.000/0.299/0.797 m/s, yaw median/p75/p95=0.003/0.284/1.050 rad/s
- match_dataset_3.jsonl: speed median/p75/p95=0.000/0.254/0.795 m/s, yaw median/p75/p95=0.001/0.210/1.014 rad/s
- match_dataset_4.jsonl: speed median/p75/p95=0.153/0.498/0.850 m/s, yaw median/p75/p95=0.123/0.575/1.167 rad/s
- match_dataset_5.jsonl: speed median/p75/p95=0.212/0.581/0.881 m/s, yaw median/p75/p95=0.176/0.624/1.184 rad/s

## First Tactical Conclusions

- The data is suitable for a first ball stop-point predictor and a first robot ETA profile.
- Use only PLAYING windows when training tactical decisions; FINISHED/READY/SET frames are useful for diagnostics but should not dominate strategy metrics.
- The first deployable tactic should be a read-only predictor in `src/tactics/`, guarded by confidence and fallback to current direct-ball logic.
- Before changing behavior, validate stop-point prediction on held-out kicks and require a median error below the robot control radius you are willing to trust.
