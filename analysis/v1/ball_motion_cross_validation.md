# Ball Motion Model Cross-Validation

This report uses leave-one-dataset-file-out cross-validation. Raw JSONL files are read-only.

## Data Filtering

- Clean free-roll tracks: 87
- Minimum travel: 0.40 m
- Minimum straightness: 0.90
- Only `ball_settled` kicks are used; the static pre-kick phase and fitted acceleration phase are excluded.
- Robot contact is approximated by straightness and monotonic deceleration filters; future recorder versions should log contact flags explicitly.

## Aggregate Held-Out Stop-Point Error

| Model | Horizon | n | Median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | 0.15s | 87 | 0.706 m | 1.468 m | 4.046 m |
| constant | 0.25s | 87 | 0.622 m | 1.209 m | 4.993 m |
| constant | 0.35s | 87 | 0.586 m | 1.338 m | 2.300 m |
| constant | 0.5s | 87 | 0.573 m | 1.082 m | 1.569 m |
| constant | 1.0s | 87 | 0.353 m | 0.789 m | 1.350 m |
| linear | 0.15s | 87 | 0.746 m | 1.435 m | 2.146 m |
| linear | 0.25s | 87 | 0.649 m | 1.349 m | 2.139 m |
| linear | 0.35s | 87 | 0.583 m | 0.991 m | 1.634 m |
| linear | 0.5s | 87 | 0.479 m | 0.793 m | 1.451 m |
| linear | 1.0s | 87 | 0.340 m | 0.579 m | 1.077 m |
| quadratic | 0.15s | 87 | 0.766 m | 1.320 m | 2.017 m |
| quadratic | 0.25s | 87 | 0.645 m | 1.252 m | 1.837 m |
| quadratic | 0.35s | 87 | 0.537 m | 0.952 m | 1.575 m |
| quadratic | 0.5s | 87 | 0.485 m | 0.789 m | 1.398 m |
| quadratic | 1.0s | 87 | 0.330 m | 0.563 m | 1.015 m |

## Fold Parameters

### Held out: `match_dataset_1.jsonl`

Train/validation tracks: 67 / 20

- constant: a0=0.4430 m/s², k=0.0000, fit samples=2375
- linear: a0=0.3114 m/s², k=0.2345, fit samples=2375
- quadratic: a0=0.3858 m/s², k=0.1290, fit samples=2375

### Held out: `match_dataset_2.jsonl`

Train/validation tracks: 69 / 18

- constant: a0=0.4443 m/s², k=0.0000, fit samples=2546
- linear: a0=0.2915 m/s², k=0.2697, fit samples=2546
- quadratic: a0=0.3774 m/s², k=0.1465, fit samples=2546

### Held out: `match_dataset_3.jsonl`

Train/validation tracks: 66 / 21

- constant: a0=0.4366 m/s², k=0.0000, fit samples=2358
- linear: a0=0.3044 m/s², k=0.2434, fit samples=2358
- quadratic: a0=0.3801 m/s², k=0.1374, fit samples=2358

### Held out: `match_dataset_4.jsonl`

Train/validation tracks: 73 / 14

- constant: a0=0.4428 m/s², k=0.0000, fit samples=2688
- linear: a0=0.2980 m/s², k=0.2535, fit samples=2688
- quadratic: a0=0.3784 m/s², k=0.1387, fit samples=2688

### Held out: `match_dataset_5.jsonl`

Train/validation tracks: 73 / 14

- constant: a0=0.4428 m/s², k=0.0000, fit samples=2677
- linear: a0=0.2926 m/s², k=0.2634, fit samples=2677
- quadratic: a0=0.3750 m/s², k=0.1462, fit samples=2677

## Decision

- Data-selected model at the 0.35s decision horizon: **quadratic**.
- 0.35s median error: 0.537 m; constant baseline: 0.586 m.
- Relative median improvement: 8.4%.
- Hard-target gates: `{'relative_improvement_at_035': False, 'median_error_at_035': True, 'p75_error_at_050': True}`.
- Approved for hard-target use: **False**.
- Recommended integration mode: **confidence_weighted_advisory**.
- Selection in this report is evidence for implementation, not automatic authorization to change match behavior.
- Before hard-target integration, require stable fold parameters, useful p75 error, and a confidence-gated fallback.
