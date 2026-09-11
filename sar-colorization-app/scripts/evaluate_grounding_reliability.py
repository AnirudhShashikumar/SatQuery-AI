#!/usr/bin/env python3
"""Evaluate GeoVision Grounding DINO reliability on a labelled local dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery_agent.grounding_evaluation import (  # noqa: E402
    DEFAULT_RELIABILITY_POLICY,
    SCORE_GRID,
    GroundingEvaluationError,
    LocalGroundingDinoCandidateProvider,
    evaluate_grounding_dataset,
    load_grounding_evaluation_dataset,
    write_grounding_evaluation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local exploratory Grounding DINO reliability evaluation; never changes production thresholds."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory containing annotations.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/grounding_eval"), help="Output artifact directory")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target", help="Evaluate only this target label")
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=DEFAULT_RELIABILITY_POLICY.minimum_alignment_score,
        help="Analysis-only reliability score gate; production configuration is unchanged",
    )
    parser.add_argument(
        "--maximum-area-ratio",
        type=float,
        default=DEFAULT_RELIABILITY_POLICY.maximum_localized_area_ratio,
        help="Analysis-only localized area gate; production configuration is unchanged",
    )
    parser.add_argument("--reuse-model", dest="reuse_model", action="store_true", default=True)
    parser.add_argument("--no-reuse-model", dest="reuse_model", action="store_false")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.minimum_score <= 1:
        parser.error("--minimum-score must be between 0 and 1.")
    if not 0 < args.maximum_area_ratio <= 1:
        parser.error("--maximum-area-ratio must be greater than 0 and at most 1.")
    try:
        dataset = load_grounding_evaluation_dataset(args.dataset)
        provider = LocalGroundingDinoCandidateProvider(
            device=args.device,
            reuse_model=args.reuse_model,
            candidate_floor=min(min(SCORE_GRID), args.minimum_score),
        )
        report = evaluate_grounding_dataset(
            dataset,
            provider,
            minimum_score=args.minimum_score,
            maximum_area_ratio=args.maximum_area_ratio,
            limit=args.limit,
            target=args.target,
        )
        artifacts = write_grounding_evaluation_outputs(report, args.output)
    except (GroundingEvaluationError, OSError) as error:
        parser.error(str(error))
    public_summary = {
        "sample_count": report["dataset"]["sample_count"],
        "metrics": report["metrics"],
        "artifact_names": artifacts,
        "production_thresholds_changed": False,
    }
    print(json.dumps(public_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
