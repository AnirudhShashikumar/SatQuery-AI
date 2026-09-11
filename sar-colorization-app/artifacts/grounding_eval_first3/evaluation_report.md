# GeoVision Local Grounding Reliability Evaluation

> Exploratory local validation only. This is not a statistically significant benchmark or scientific performance claim.

## Dataset

- Samples: 3
- Positive samples: 3
- Negative samples: 0
- Categories: {"harbor": 1, "vehicle": 2}
- Warning: Results are exploratory and not statistically significant for a small local dataset.

## Model and operational gate

- Checkpoint: IDEA-Research/grounding-dino-tiny
- Device(s): mps
- Production minimum alignment score: 0.45
- Production maximum localized area ratio: 0.85
- Production processor candidate threshold: 0.35
- Evaluation candidate floor: 0.3
- Disclaimer: Operational reliability gates pending benchmark calibration.
- Alignment scores are text-region alignment scores, not probabilities.

## Evaluation-gate metrics

- Accepted Precision Iou 25: 0.0000
- Accepted Recall Iou 25: 0.0000
- Accepted Precision Iou 50: 0.0000
- Accepted Recall Iou 50: 0.0000
- False Positive Count: 0
- False Localization Count: 1
- Missed Target Count: 3
- No Prediction Count: 0
- Mean Accepted Best Iou: 0.0811
- Median Accepted Best Iou: 0.0811
- Acceptance Rate: 0.1667
- Rejection Rate: 0.8333
- Full Frame Candidate Rate: 0.0000
- Score distribution: {"count": 6, "maximum": 0.459628, "mean": 0.36638796, "median": 0.355528725, "minimum": 0.32341439}
- Box area ratio distribution: {"count": 6, "maximum": 0.87157702, "mean": 0.35850839166666665, "median": 0.31970126, "minimum": 0.00433487}

These fields distinguish raw model-candidate distributions, operational gate acceptance/rejection, and reference-box localization accuracy.

## IoU definitions

- IoU 0.25: a matched prediction/reference pair has IoU >= 0.25.
- IoU 0.50: a matched prediction/reference pair has IoU >= 0.50.
- Aggregate precision and recall use greedy one-to-one matching by descending IoU.

## Metrics by category

| Target | Samples | Precision @ 0.25 | Recall @ 0.25 | Missed targets |
|---|---:|---:|---:|---:|
| harbor | 1 | 0.0000 | 0.0000 | 1 |
| vehicle | 2 | Unavailable | 0.0000 | 2 |

## Auditable examples

- Reliable Localization: Unavailable
- Low Score Rejection: vehicle_p0003_0002_q0, vehicle_p0003_0004_q4, harbor_p0019_0002_q21
- Near Full Frame Rejection: Unavailable
- False Accepted Localization: harbor_p0019_0002_q21
- Missed Target: vehicle_p0003_0002_q0, vehicle_p0003_0004_q4, harbor_p0019_0002_q21

## Threshold grid

- Evaluated combinations: 36
- Grid values are analysis-only. No production threshold was selected or changed.
- Candidate values should not be recommended until enough representative labelled samples are available.

## Limitations

- Grounding DINO is a general-domain zero-shot detector and is not calibrated for remote-sensing imagery.
- Small local datasets do not support statistically significant performance claims.
- Reference-box quality, target scale, sensor characteristics, and scene diversity directly affect these metrics.
- IoU measures box overlap; it does not establish scientific or semantic ground truth beyond the supplied annotations.
