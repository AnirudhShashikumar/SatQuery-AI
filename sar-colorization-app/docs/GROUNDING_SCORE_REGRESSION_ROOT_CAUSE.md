# Grounding score regression root cause

## Finding

Grounding DINO proposal generation, official post-processing, boxes, and labels remain correct. The regression occurs after official post-processing in `RemoteSensingGrounder.ground()`:

1. Grounding DINO produces token-alignment logits and proposal boxes.
2. `post_process_grounded_object_detection()` returns the accepted proposal boxes, labels, and Grounding DINO scores.
3. Grounding Specialist v1.1 runs its query-scoring head over `last_hidden_state` and applies sigmoid.
4. `specialist_scores_for_accepted_proposals()` maps those specialist probabilities to the proposals retained by the official postprocessor.
5. The mapped values replace the Grounding DINO scores unconditionally.
6. `evaluate_detection_quality()` applies the existing `minimum_alignment_score=0.45` gate to the replacement values.

The specialist probabilities are not calibrated to Grounding DINO text-region alignment scores. Observed specialist values around `0.005–0.05` therefore fail a reliability threshold designed for a different score distribution. This suppresses candidates without improving or changing their boxes.

## Required correction

Grounding DINO scores must remain authoritative in production. Specialist outputs need an explicit score policy, distribution validation, and a preservation check. The default fallback policy must revert to the original Grounding DINO scores whenever specialist scoring is missing, invalid, extremely small, distributionally unsuitable, or changes the accepted geometry/label set.

No checkpoint, proposal, box, label, NMS, processor, tokenizer, prompt, route, or API contract needs to change.
