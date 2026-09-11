# RSVQA Smoke Repair Priorities

Priority score is `affected samples × expected recoverability`. Recoverability is an explicit diagnostic estimate, not measured improvement.
No recommendation uses ground-truth metadata at inference time or hard-codes benchmark answers.

## Routing/Normalization Fixes

### 3. Cover benchmark presence constructions that fell through to generic or unsupported routing

- Affected samples: 4
- Expected recoverability: 0.90
- Impact score: 3.60
- Recommendation: Add text-only routing tests for observed 'Is a … present' forms while retaining centralized answer normalization.

## Evidence-Availability Fixes

### 1. Expand controlled target/evidence availability for missing operands and vocabulary

- Affected samples: 11
- Expected recoverability: 0.75
- Impact score: 8.25
- Recommendation: Add auditable support for currently unresolved target terms and require both comparison operands; do not use benchmark labels or metadata at inference time.

### 2. Treat zero accepted grounding regions as uncertain evidence, not an automatically reliable zero count

- Affected samples: 10
- Expected recoverability: 0.60
- Impact score: 6.00
- Recommendation: Calibrate zero-region behavior and distinguish no detection from confirmed absence before deriving counts or comparisons.

## Model-Quality Limitations

### 4. Improve rural/urban and presence discrimination after routing succeeds

- Affected samples: 6
- Expected recoverability: 0.40
- Impact score: 2.40
- Recommendation: Evaluate calibrated or retrained scene/presence evidence on a held-out set; the smoke result does not justify claiming improvement.

## Benchmark-Specific Risks

### 5. Resolve the mismatch between dense RSVQA count labels and accepted-region box counts

- Affected samples: 9
- Expected recoverability: 0.15
- Impact score: 1.35
- Recommendation: Determine whether a scientifically valid counting model can represent dense road/building/area counts before a full run; never hard-code answers.

### 6. Investigate unclassified failures only after richer evidence is persisted

- Affected samples: 0
- Expected recoverability: 0.25
- Impact score: 0.00
- Recommendation: Persist safe operand/count evidence in future benchmark runs; do not infer causes from absent fields.

## Recommended Decision

**B. Fix routing/evidence first.** The run has 30% null coverage loss and identifiable routing, vocabulary, and operand-availability gaps. Resolve those before paying for a 10,004-question run or attributing residual errors to model quality.
