# PE evaluation results

The following two PE groups are saved from the original evaluation runs under `pe_data/testset_110/runs/`. Each run directory contains its candidate records, evidence, per-case judgments, run metadata, and generated summary.

| PE group | Modality run | Cases | Status | Failed stage files |
| --- | --- | ---: | --- | ---: |
| `weak-pe-original` | `weak-pe-testset110-t2v-20261113` | 50 | completed with errors | 4 |
| `weak-pe-original` | `weak-pe-testset110-i2v-r2v-20261113` | 60 | completed with errors | 3 |
| `glm53_20261112_120702` | `glm-pe-testset110-t2v-20261113` | 50 | completed with errors | 3 |
| `glm53_20261112_120702` | `glm-pe-testset110-i2v-r2v-20261113` | 60 | completed | 0 |

Benchmark records and media assets are excluded from the repository. The saved runs retain their dataset fingerprints and original metadata; reproducing them requires the matching benchmark data available separately.
