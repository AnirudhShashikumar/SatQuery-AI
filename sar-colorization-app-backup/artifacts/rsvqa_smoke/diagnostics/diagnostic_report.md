# RSVQA-LR Smoke Diagnostic Report

This report analyzes the persisted 50-question smoke run only. It does not change production behavior and does not claim a full RSVQA benchmark result.

## Aggregate Verification

All supplied aggregate checks passed: **True**.

- Samples: 50
- Answered: 35
- Null: 15
- Coverage: 70.00%
- All-sample accuracy: 38.00%
- Answered-only accuracy: 54.29%

## Metrics by Type

| Type | Samples | Coverage | Null rate | Overall accuracy | Answered-only accuracy | Majority baseline |
|---|---:|---:|---:|---:|---:|---:|
| comp | 15 | 60.00% | 40.00% | 40.00% | 66.67% | no (73.33%) |
| count | 15 | 86.67% | 13.33% | 40.00% | 46.15% | 0 (40.00%) |
| presence | 15 | 53.33% | 46.67% | 33.33% | 62.50% | yes (60.00%) |
| rural_urban | 5 | 100.00% | 0.00% | 40.00% | 40.00% | urban (80.00%) |

No question type exceeds its majority-class baseline in this smoke sample. Count accuracy equals its zero-label baseline; it does not exceed it.

## Confidence and Latency

| Outcome | Samples | Average confidence | Average latency (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|
| correct | 19 | 0.105263 | 1141.684 | 1313.0 | 2211.0 |
| wrong | 16 | 0.140625 | 1275.438 | 1144.0 | 2961.0 |
| null | 15 | 0.25 | 727.867 | 474.0 | 1340.3 |

### Latency by task

| Task | Samples | Average (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| comparison_vqa | 15 | 1902.267 | 1786.0 | 3087.0 |
| count_vqa | 15 | 1137.333 | 1317.0 | 1350.3 |
| presence_vqa | 11 | 363.727 | 465.0 | 481.0 |
| rural_urban_classification | 5 | 327.4 | 457.0 | 546.8 |
| unsupported | 2 | 425.0 | 425.0 | 436.7 |
| vqa | 2 | 467.5 | 467.5 | 473.35 |

## Failure Taxonomy

| Primary cause | Samples |
|---|---:|
| grounding returned zero regions | 10 |
| comparison operand unavailable | 6 |
| model prediction error | 6 |
| unsupported target or vocabulary | 5 |
| routing failure | 4 |

Causal labels are conservative. In particular, raw comparison operand counts, caption scores, and SVE consistency objects were not persisted, so absent evidence is classified as `unknown` rather than inferred.

## Wording and Target Findings

- `comp / equal/same`: n=3, coverage=33.33%, accuracy=0.00%, null=66.67%.
- `comp / less/fewer`: n=6, coverage=50.00%, accuracy=33.33%, null=50.00%.
- `comp / more/greater`: n=6, coverage=83.33%, accuracy=66.67%, null=16.67%.
- `count / amount of`: n=1, coverage=100.00%, accuracy=0.00%, null=0.00%.
- `count / how many`: n=6, coverage=66.67%, accuracy=50.00%, null=33.33%.
- `count / number of`: n=7, coverage=100.00%, accuracy=28.57%, null=0.00%.
- `count / relational count`: n=1, coverage=100.00%, accuracy=100.00%, null=0.00%.
- `presence / is there`: n=11, coverage=72.73%, accuracy=45.45%, null=27.27%.
- `presence / visible/can you see`: n=4, coverage=0.00%, accuracy=0.00%, null=100.00%.
- `rural_urban / rural-or-urban`: n=5, coverage=100.00%, accuracy=40.00%, null=0.00%.

Target-level metrics are in `metrics_by_target.csv`; comparison rows contribute to both entity targets.

## Answer Distributions

- **comp:** ground truth {'no': 11, 'yes': 4}; predictions {'no': 8, 'yes': 1}; null=6; most frequent prediction={'answer': 'no', 'count': 8}; majority baseline=no at 73.33%.
- **count:** ground truth {'0': 6, '10': 1, '2': 1, '20': 1, '246': 1, '27': 1, '33': 1, '3428': 1, '4': 1, '8': 1}; predictions {'0': 13}; null=2; most frequent prediction={'answer': '0', 'count': 13}; majority baseline=0 at 40.00%.
- **presence:** ground truth {'no': 6, 'yes': 9}; predictions {'yes': 8}; null=7; most frequent prediction={'answer': 'yes', 'count': 8}; majority baseline=yes at 60.00%.
- **rural_urban:** ground truth {'rural': 1, 'urban': 4}; predictions {'rural': 4, 'urban': 1}; null=0; most frequent prediction={'answer': 'rural', 'count': 4}; majority baseline=urban at 80.00%.

### Count label buckets

- ground truth 0: n=6, coverage=100.00%, overall accuracy=100.00%.
- ground truth 1: n=0, coverage=n/a, overall accuracy=n/a.
- ground truth 2+: n=9, coverage=77.78%, overall accuracy=0.00%.
- Zero prediction dominance: 13/13 answered count questions (100.00%) predicted `0`.

## Priority Repairs

1. **Expand controlled target/evidence availability for missing operands and vocabulary** — affected=11, recoverability=0.75, impact=8.25.
2. **Treat zero accepted grounding regions as uncertain evidence, not an automatically reliable zero count** — affected=10, recoverability=0.60, impact=6.00.
3. **Cover benchmark presence constructions that fell through to generic or unsupported routing** — affected=4, recoverability=0.90, impact=3.60.
4. **Improve rural/urban and presence discrimination after routing succeeds** — affected=6, recoverability=0.40, impact=2.40.
5. **Resolve the mismatch between dense RSVQA count labels and accepted-region box counts** — affected=9, recoverability=0.15, impact=1.35.
6. **Investigate unclassified failures only after richer evidence is persisted** — affected=0, recoverability=0.25, impact=0.00.

## Full 10,004-Question Projection

**Projection, not a measured benchmark.** It assumes smoke-run rates and sequential latency remain unchanged.

- Expected correct: 3801.52 (~3802)
- Expected null: 3001.20 (~3001)
- Mean-latency runtime: 2.947 hours
- P95-rate runtime scenario: 6.207 hours

## Decision

**B. Fix routing/evidence first.** The current 30% null rate and recoverable routing/vocabulary/operand gaps make a full run premature. Reassess model training or replacement after those deterministic gaps are removed on a held-out smoke set.
