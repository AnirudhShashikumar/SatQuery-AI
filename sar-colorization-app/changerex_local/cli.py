"""Command-line interface for standalone ChangerEx inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import DEFAULT_MAXIMUM_DIMENSION, DEFAULT_THRESHOLD
from .diagnostics import benchmark_summary, save_artifacts
from .inference import ChangerExInferenceError, predict_change
from .lifecycle import configure_lifecycle
from .preprocessing import load_rgb_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone pure-PyTorch ChangerEx inference")
    parser.add_argument("--earlier", required=True, help="Earlier RGB image")
    parser.add_argument("--later", required=True, help="Later RGB image")
    parser.add_argument("--checkpoint", required=True, help="Official ChangerEx checkpoint")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--allow-device-fallback", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument("--maximum-dimension", type=int, default=DEFAULT_MAXIMUM_DIMENSION)
    parser.add_argument("--save-probability-npy", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    if args.maximum_dimension <= 0:
        raise ValueError("--maximum-dimension must be positive")
    if args.warmup_runs < 0 or args.benchmark_runs < 0:
        raise ValueError("--warmup-runs and --benchmark-runs cannot be negative")


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    configure_lifecycle(
        args.checkpoint,
        device=args.device,
        allow_device_fallback=args.allow_device_fallback,
    )
    prediction_kwargs = {
        "device": args.device,
        "threshold": args.threshold,
        "maximum_dimension": args.maximum_dimension,
        "allow_device_fallback": args.allow_device_fallback,
    }
    result = predict_change(args.earlier, args.later, **prediction_kwargs)
    first_runtime = result.runtime
    first_was_reused = result.load_reuse_status.get("was_reused")
    baseline_probability = result.probability_map.copy()
    for _ in range(args.warmup_runs):
        result = predict_change(args.earlier, args.later, **prediction_kwargs)
    benchmark_times: list[float] = []
    repeat_difference: float | None = None
    for _ in range(args.benchmark_runs):
        result = predict_change(args.earlier, args.later, **prediction_kwargs)
        assert result.runtime is not None
        benchmark_times.append(result.runtime.model_seconds)
        repeat_difference = float(np.max(np.abs(result.probability_map - baseline_probability)))
    benchmark = benchmark_summary(
        benchmark_times, repeat_max_abs_difference=repeat_difference
    )
    benchmark.update(
        {
            "first_total_seconds": first_runtime.inference_seconds if first_runtime else None,
            "first_model_seconds": first_runtime.model_seconds if first_runtime else None,
            "first_was_reused": first_was_reused,
        }
    )
    later = load_rgb_image(args.later)
    paths = save_artifacts(
        result,
        later,
        args.output_dir,
        benchmark=benchmark,
        # Required artifacts always include the NPY. The flag remains accepted
        # for an explicit caller contract and backward-compatible scripting.
        save_probability_npy=True,
    )
    summary = result.summary()
    summary["benchmark"] = benchmark
    summary["artifacts"] = paths
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        summary = run(args)
        print(json.dumps(summary, indent=2))
        return 0
    except (ChangerExInferenceError, ValueError, RuntimeError, OSError) as error:
        print(
            json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"[:1000]}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
