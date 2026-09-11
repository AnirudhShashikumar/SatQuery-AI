# RSVQA Smoke Benchmark: v1 vs v2

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Accuracy | 38.00% | 22.00% | -16.00% |
| Coverage | 70.00% | 40.00% | -30.00% |
| Null answers | 15 | 30 | +15 |
| Endpoint errors | 0 | 0 | +0 |

Fixed questions: **5**. Regressions: **13**.

## Per-type results

| Type | v1 accuracy | v2 accuracy | v1 coverage | v2 coverage | Majority baseline | v2 exceeds |
|---|---:|---:|---:|---:|---:|:---:|
| comp | 40.00% | 6.67% | 60.00% | 6.67% | no (73.33%) | no |
| count | 40.00% | 0.00% | 86.67% | 33.33% | 0 (40.00%) | no |
| presence | 33.33% | 46.67% | 53.33% | 73.33% | yes (60.00%) | no |
| rural_urban | 40.00% | 60.00% | 100.00% | 60.00% | urban (80.00%) | no |

## Count prediction distribution

- v1: `{'0': 13, 'null': 2}`
- v2: `{'1': 5, 'null': 10}`

Counts are accepted localization regions, not calibrated object totals.
