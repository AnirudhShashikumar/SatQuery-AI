import json
from argparse import Namespace

import numpy as np
from PIL import Image

import changerex_local.cli as cli_module
from changerex_local.cli import main, run
from changerex_local.schemas import ChangerExResult, PreprocessingDetails, RuntimeStats


def _result():
    probability = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    mask = (probability >= 0.5).astype(np.uint8)
    return ChangerExResult(
        source_width=2, source_height=2, model_input_width=32, model_input_height=32,
        selected_device="cpu", probability_map=probability, binary_mask=mask,
        changed_pixel_count=2, changed_percentage=50.0, threshold=0.5,
        preprocessing=PreprocessingDetails(
            (2, 2), (2, 2), (32, 32), 1.0, "RGB", "earlier RGB, then later RGB",
            "[0,255]", (1, 2, 3), (4, 5, 6), "bilinear", (0, 0, 30, 30), 32,
        ),
        runtime=RuntimeStats(0.1, 0.2, 0.01, 0.1, 0.09, 10.0),
        load_reuse_status={"was_reused": False, "load_count": 1, "reuse_count": 0, "inference_count": 1, "state": "ready"},
    )


def test_cli_outputs_all_artifacts(monkeypatch, tmp_path):
    earlier = tmp_path / "a.png"
    later = tmp_path / "b.png"
    Image.new("RGB", (2, 2)).save(earlier)
    Image.new("RGB", (2, 2)).save(later)
    monkeypatch.setattr(cli_module, "configure_lifecycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "predict_change", lambda *args, **kwargs: _result())
    output = tmp_path / "output"
    args = Namespace(
        earlier=str(earlier), later=str(later), checkpoint="test.pt", output_dir=str(output),
        device="cpu", threshold=0.5, allow_device_fallback=False, warmup_runs=0,
        benchmark_runs=1, maximum_dimension=32, save_probability_npy=True,
    )
    summary = run(args)
    for name in (
        "probability_map.npy", "probability_map.png", "binary_mask.png", "display_mask.png",
        "overlay.png", "regions.json", "result.json", "checkpoint_verification.json",
        "environment.json", "benchmark_report.md",
    ):
        assert (output / name).is_file()
    assert summary["selected_device"] == "cpu"


def test_cli_failure_is_nonzero_json(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "run", lambda args: (_ for _ in ()).throw(ValueError("failure")))
    code = main(["--earlier", "a", "--later", "b", "--checkpoint", "c", "--output-dir", "d"])
    assert code == 1
    assert json.loads(capsys.readouterr().err)["ok"] is False
