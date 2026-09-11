#!/usr/bin/env python3
"""Send one image/question pair to the benchmarkable SatQuery VQA endpoint."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from pathlib import Path
from typing import Any

import requests


def _print_compact_summary(payload: dict[str, Any], status_code: int) -> None:
    """Print the benchmark fields without dumping the full response envelope."""
    print(f"HTTP status: {status_code}")
    print(f"answer: {payload.get('answer')}")
    print(f"task: {payload.get('task')}")
    print(f"confidence: {payload.get('confidence')}")
    print(f"status: {payload.get('status') or payload.get('result_status')}")
    print(f"processing time: {payload.get('processing_time_ms')} ms")


def _print_http_error(response: requests.Response, payload: Any, *, full_json: bool) -> None:
    """Prefer the server's structured error over requests' exception text."""
    print(f"HTTP {response.status_code} {response.reason}", file=sys.stderr)
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2 if full_json else None, ensure_ascii=False), file=sys.stderr)
    else:
        print(str(payload), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG, JPEG, or WEBP image to analyze")
    parser.add_argument("question", help="RSVQA question, for example: Is there a river?")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Backend base URL")
    parser.add_argument("--analysis-type", default="ground_truth", help="Existing image-analysis type")
    parser.add_argument(
        "--model-name",
        default="satquery-agent",
        help="Compatibility form value; question requests execute through SatQuery",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON response")
    parser.add_argument("--verbose", action="store_true", help="Print a traceback when the request fails")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    mime_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    try:
        with args.image.open("rb") as image_file:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/api/analysis/image",
                data={
                    "analysis_type": args.analysis_type,
                    "model_name": args.model_name,
                    "question": args.question,
                },
                files={"image": (args.image.name, image_file, mime_type)},
                timeout=args.timeout,
            )
    except requests.RequestException as error:
        print(f"Request failed: {error}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    try:
        payload = response.json()
    except ValueError:
        payload = response.text or "The server returned an empty non-JSON response."

    if not response.ok:
        _print_http_error(response, payload, full_json=args.json)
        if args.verbose:
            try:
                response.raise_for_status()
            except requests.HTTPError:
                traceback.print_exc()
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif isinstance(payload, dict):
        _print_compact_summary(payload, response.status_code)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
