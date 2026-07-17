# Robot ETA Team2 Candidate Evaluation

This evaluation keeps the deployed candidate unchanged and treats the newly
trained eight-file model as a challenger. All training validation holds out one
complete JSONL file, so suffix queries from the same ETA episode cannot cross
the train/test boundary.

## Training data

- Existing five match files
- `dataset/v4/ETA/match_dataset.team_new.jsonl`
- Two files from `dataset/v4/ETA/match_dataset.team2`
- 277 completed travel episodes and 4,726 fixed-interval ETA queries

## Eight-file challenger validation

The selected constant-time model is `balanced_ridge_0.1`.

| Query point | n | Median absolute error | P75 | P90 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| All suffix queries | 4,726 | 0.486 s | 0.891 s | 1.584 s | 0.796 s |
| Original episode starts | 277 | 0.688 s | 1.371 s | 2.439 s | 1.152 s |

Prediction runtime is 8.0 microseconds at the median and 26.5 microseconds at
P99 in the local benchmark.

## Fair comparison on the three new files

For the challenger, each new file is predicted by a model trained on the other
seven files. The existing model was trained before all three files and is also
fully out of sample on this subset.

| Model | Median | P75 | P90 | Start median | Start P75 | Start P90 | Composite score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing model | 0.497 s | 0.871 s | 2.003 s | 0.748 s | 1.491 s | 2.703 s | 1.797 |
| Eight-file challenger | 0.542 s | 0.982 s | 1.883 s | 0.769 s | 1.526 s | 2.822 s | 1.869 |

The challenger reduces the all-query P90 tail by 0.120 seconds, but regresses
the median, P75, start P90, and the composite score. Therefore it is retained as
a trained challenger and must not replace `robot_eta_model.json` yet. More data
alone is unlikely to fix this result; the next model iteration should add
player/controller-aware structure or a gated tail correction and prove an
improvement on whole-file-held-out validation before deployment.
