# RSVQA Smoke Benchmark: Old Heuristic vs RSVQA Specialist v1

| Metric | Old heuristic | RSVQA Specialist v1 | Change |
|---|---:|---:|---:|
| Accuracy | 38.00% | 80.00% | +42.00% |
| Coverage | 70.00% | 100.00% | +30.00% |
| Null answers | 15 | 0 | -15 |
| Endpoint errors | 0 | 0 | +0 |

Fixed questions: **21**. Regressions: **0**.

## Per-type results

| Type | Heuristic accuracy | Specialist accuracy | Heuristic coverage | Specialist coverage | Majority baseline | Specialist exceeds |
|---|---:|---:|---:|---:|---:|:---:|
| comp | 40.00% | 86.67% | 60.00% | 100.00% | no (73.33%) | yes |
| count | 40.00% | 53.33% | 86.67% | 100.00% | 0 (40.00%) | yes |
| presence | 33.33% | 93.33% | 53.33% | 100.00% | yes (60.00%) | yes |
| rural_urban | 40.00% | 100.00% | 100.00% | 100.00% | urban (80.00%) | yes |

## Latency

| Metric | Old heuristic | RSVQA Specialist v1 |
|---|---:|---:|
| Average | 1060.3 ms | 164.0 ms |
| Median | 949.5 ms | 203.0 ms |
| P95 | 2233.8 ms | 212.8 ms |

## Count prediction distribution

- Old heuristic: `{'0': 13, 'null': 2}`
- RSVQA Specialist v1: `{'0': 7, '201+': 2, '3': 1, '4': 1, '40': 1, '53': 1, '8': 1, '9': 1}`

The specialist's `201+` class is the exported overflow label. Counts are learned benchmark labels, not calibrated physical object totals.

## Confusion matrices

Combined old/new confusion matrices are in `confusion_matrices_overall.csv` and `confusion_matrices_by_task.csv`.
