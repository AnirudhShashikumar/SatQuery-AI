#!/usr/bin/env python3
"""Reference-caption evaluation for the existing SatQuery caption specialist."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    from evaluation_common import benchmark_record, write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import benchmark_record, write_csv, write_json


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def bleu_n(candidate: str, references: list[str], n: int) -> float:
    values = tokens(candidate)
    if len(values) < n:
        return 0.0
    candidate_ngrams = Counter(tuple(values[index:index + n]) for index in range(len(values) - n + 1))
    maximum = Counter()
    for reference in references:
        ref = tokens(reference)
        grams = Counter(tuple(ref[index:index + n]) for index in range(max(0, len(ref) - n + 1)))
        maximum |= grams
    clipped = sum(min(count, maximum[gram]) for gram, count in candidate_ngrams.items())
    precision = clipped / sum(candidate_ngrams.values())
    closest = min((len(tokens(value)) for value in references), key=lambda length: abs(length - len(values)))
    brevity = 1.0 if len(values) > closest else math.exp(1.0 - closest / max(len(values), 1))
    return brevity * precision


def rouge_l(candidate: str, references: list[str]) -> float:
    left = tokens(candidate)
    best = 0.0
    for reference in references:
        right = tokens(reference)
        table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        for i, a in enumerate(left, 1):
            for j, b in enumerate(right, 1):
                table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
        lcs = table[-1][-1]
        precision, recall = lcs / max(len(left), 1), lcs / max(len(right), 1)
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, score)
    return best


def evaluate(args: argparse.Namespace) -> dict:
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Caption manifest with references is unavailable: {manifest}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or any(not item.get("references") for item in records):
        raise ValueError("Every caption record must contain one or more reference captions")
    root, output = args.dataset_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    rows, failures = [], []
    for index, record in enumerate(records):
        image = root / record["image"]
        if not image.is_file():
            raise FileNotFoundError(f"Missing caption image: {image}")
        try:
            response = client.post("/api/agent/query", data={"query": "Describe the land-cover and major objects visible in this image.", "input_mode": "single", "primary_modality": "optical", "use_cache": "false"}, files={"primary_image": (image.name, image.read_bytes())})
            response.raise_for_status()
            prediction = response.json().get("answer") or ""
            references = [str(value) for value in record["references"]]
            rows.append({"sample_id": record.get("id", index), "prediction": prediction, **{f"bleu_{n}": bleu_n(prediction, references, n) for n in range(1, 5)}, "rouge_l": rouge_l(prediction, references)})
        except Exception as error:
            failures.append({"sample_id": record.get("id", index), "error": f"{type(error).__name__}: {error}"[:500]})
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "failures.csv", failures, ("sample_id", "error"))
    summary = {"samples": len(records), "completed": len(rows), "failures": len(failures), "metrics": {name: sum(row[name] for row in rows) / len(rows) if rows else None for name in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l")}, "meteor": None, "cider": None, "status": "verified_test" if len(rows) == len(records) and not failures else "unavailable"}
    write_json(output / "summary.json", summary)
    metric_rows = [
        {"name": name, "value": value, "unit": "score", "primary": name in {"bleu_4", "rouge_l"}, "definition": "Corpus mean of per-image reference-caption score."}
        for name, value in summary["metrics"].items()
    ] + [
        {"name": "meteor", "value": None, "unit": "score", "primary": False, "definition": "Unavailable: no approved implementation bundled."},
        {"name": "cider", "value": None, "unit": "score", "primary": False, "definition": "Unavailable: no approved implementation bundled."},
    ]
    write_json(output / "benchmark_record.json", benchmark_record(
        benchmark_id="caption.reference.v1", specialist_id="rs_captioner",
        display_name="Reference caption evaluation", task="captioning",
        model_name="SatQuery remote-sensing captioner", model_version="production",
        checkpoint=None, checkpoint_sha256=None, dataset=manifest.stem, split="test",
        sample_count=len(records) if summary["status"] == "verified_test" else None,
        status=summary["status"], metrics=metric_rows, performance={},
        artifacts=["predictions.csv", "failures.csv", "summary.json", "report.md"],
        limitations=["METEOR and CIDEr are unavailable until an approved implementation is connected.", "Scores require supplied reference captions."],
    ))
    (output / "report.md").write_text("# Caption evaluation\n\n" + json.dumps(summary, indent=2) + "\n\nMETEOR and CIDEr are not claimed because no approved implementation is bundled.\n", encoding="utf-8")
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/caption_test"))
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
