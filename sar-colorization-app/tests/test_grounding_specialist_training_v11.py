"""Focused tests for the Grounding Specialist v1.1 training recipe."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, SequentialSampler, WeightedRandomSampler

from training import cli
from training import grounding_specialist_v11 as training


class LabelDataset(Dataset[int]):
    def __init__(self) -> None:
        self.class_labels = ["vehicle"] * 13 + ["container-crane"]

    def __len__(self) -> int:
        return len(self.class_labels)

    def __getitem__(self, index: int) -> int:
        return index


def optimizer_and_scheduler(head: torch.nn.Module) -> tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = training.make_cosine_scheduler(
        optimizer,
        total_steps=10,
        minimum_learning_rate_ratio=0.1,
    )
    return optimizer, scheduler


def checkpoint_with_training_state(path: Path, *, step: int = 4) -> training.SatQueryGroundingHead:
    head = training.SatQueryGroundingHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = training.make_cosine_scheduler(
        optimizer,
        total_steps=100,
        minimum_learning_rate_ratio=0.1,
    )
    for _ in range(step):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in head.parameters()).backward()
        optimizer.step()
        scheduler.step()
    payload = training.checkpoint_payload(
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        history=[{"step": value} for value in range(1, step + 1)],
        validation={"accuracy_at_050": 0.4},
        configuration={
            "completed_passes": 1,
            "validation_history": [{"global_step": step, "accuracy_at_050": 0.4}],
            "early_stopping_best_accuracy50": 0.4,
            "validations_without_improvement": 1,
        },
        area_regularization_weight=0.05,
        query_entropy_weight=0.01,
    )
    training.save_checkpoint(path, payload)
    return head


def test_weighted_sampler_is_active_only_for_training_and_has_equal_class_mass() -> None:
    dataset = LabelDataset()
    train_loader = training.build_data_loader(dataset, batch_size=2, training=True)
    validation_loader = training.build_data_loader(dataset, batch_size=2, training=False)

    assert isinstance(train_loader.sampler, WeightedRandomSampler)
    assert train_loader.sampler.replacement is True
    assert train_loader.sampler.num_samples == len(dataset)
    assert isinstance(validation_loader.sampler, SequentialSampler)

    class_mass: defaultdict[str, float] = defaultdict(float)
    for label, weight in zip(dataset.class_labels, train_loader.sampler.weights.tolist()):
        class_mass[label] += weight
    assert class_mass["vehicle"] == pytest.approx(class_mass["container-crane"])


def test_area_regularization_loss_is_finite_and_backpropagates() -> None:
    predicted = torch.tensor(
        [[0.5, 0.5, 0.8, 0.6], [0.3, 0.4, 0.2, 0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([[0.3, 0.35, 0.7, 0.65], [0.2, 0.3, 0.4, 0.5]])
    loss = training.box_area_regularization_loss(predicted, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_query_entropy_loss_is_finite_bounded_and_rewards_diversity() -> None:
    diverse_logits = torch.zeros(4, 8, requires_grad=True)
    collapsed_logits = torch.full((4, 8), -8.0)
    collapsed_logits[:, 0] = 8.0
    diversity_loss, entropy = training.query_diversity_loss(diverse_logits)
    collapsed_loss, collapsed_entropy = training.query_diversity_loss(collapsed_logits)

    assert torch.isfinite(diversity_loss)
    assert torch.isfinite(entropy)
    assert 0.0 <= diversity_loss.item() <= 1.0
    assert collapsed_loss.item() > diversity_loss.item()
    assert collapsed_entropy.item() < entropy.item()
    diversity_loss.backward()
    assert diverse_logits.grad is not None


def test_combined_loss_preserves_pilot_weights_and_logs_v11_terms() -> None:
    torch.manual_seed(4)
    logits = torch.randn(2, 6, requires_grad=True)
    base_boxes = torch.rand(2, 6, 4)
    base_boxes[..., 2:] = base_boxes[..., 2:] * 0.4 + 0.05
    refined_boxes = base_boxes.detach().clone().requires_grad_(True)
    targets = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.8]])
    losses = training.grounding_specialist_losses(
        logits,
        refined_boxes,
        base_boxes,
        targets,
        area_regularization_weight=0.05,
        query_entropy_weight=0.01,
    )
    expected = (
        losses["classification_loss"]
        + 5.0 * losses["l1_loss"]
        + 2.0 * losses["giou_loss"]
        + 0.05 * losses["area_loss"]
        + 0.01 * losses["diversity_loss"]
    )
    assert torch.allclose(losses["loss"], expected)
    assert torch.isfinite(losses["area_loss"])
    assert torch.isfinite(losses["diversity_loss"])
    losses["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(refined_boxes.grad).all()


def test_v11_checkpoint_keeps_legacy_format_and_strict_head_layout(tmp_path: Path) -> None:
    head = training.SatQueryGroundingHead()
    optimizer, scheduler = optimizer_and_scheduler(head)
    payload = training.checkpoint_payload(
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        step=12,
        history=[{"step": 12, "loss": 1.0}],
        validation={"accuracy_at_050": 0.5},
        configuration={"batch_size": 8},
        area_regularization_weight=0.05,
        query_entropy_weight=0.01,
    )
    legacy_keys = {
        "model_name",
        "step",
        "base_model",
        "specialist_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "history",
        "validation",
        "configuration",
    }
    assert legacy_keys.issubset(payload)
    assert payload["training_recipe_version"] == "v1.1"
    assert payload["balanced_sampling"] is True
    assert payload["area_regularization_weight"] == 0.05
    assert payload["query_entropy_weight"] == 0.01

    path = tmp_path / "v11.pt"
    training.save_checkpoint(path, payload)
    deployed_head = training.SatQueryGroundingHead()
    loaded = training.load_training_checkpoint(path, deployed_head)
    assert loaded.step == 12
    assert list(deployed_head.state_dict()) == [
        "query_scorer.0.weight",
        "query_scorer.0.bias",
        "query_scorer.1.weight",
        "query_scorer.1.bias",
        "query_scorer.4.weight",
        "query_scorer.4.bias",
        "box_refiner.0.weight",
        "box_refiner.0.bias",
        "box_refiner.1.weight",
        "box_refiner.1.bias",
        "box_refiner.4.weight",
        "box_refiner.4.bias",
    ]
    for key, value in head.state_dict().items():
        assert torch.equal(value, deployed_head.state_dict()[key])


def test_old_checkpoint_without_v11_metadata_still_loads(tmp_path: Path) -> None:
    source = training.SatQueryGroundingHead()
    legacy = {
        "model_name": "SatQuery Grounding Specialist v1 Pilot",
        "step": 100,
        "base_model": training.BASE_CHECKPOINT,
        "specialist_state_dict": source.state_dict(),
        "history": [{"step": 100, "loss": 1.2}],
        "validation": {"accuracy_at_050": 0.25},
        "configuration": {"batch_size": 8},
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    destination = training.SatQueryGroundingHead()
    loaded = training.load_training_checkpoint(path, destination)

    assert loaded.step == 100
    assert loaded.metadata == {
        "training_recipe_version": "v1.0",
        "balanced_sampling": False,
        "area_regularization_weight": 0.0,
        "query_entropy_weight": 0.0,
    }
    for key, value in source.state_dict().items():
        assert torch.equal(value, destination.state_dict()[key])


def test_weights_only_resume_resets_optimizer_scheduler_history_and_global_step(tmp_path: Path) -> None:
    path = tmp_path / "source.pt"
    source = checkpoint_with_training_state(path, step=4)
    destination = training.SatQueryGroundingHead()
    optimizer, scheduler = optimizer_and_scheduler(destination)

    restored = cli.restore_run_state(
        path,
        resume_optimizer=False,
        head=destination,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert restored.mode == "weights_only"
    assert restored.global_step == 0
    assert restored.history == []
    assert restored.validation_history == []
    assert restored.source_checkpoint_step == 4
    assert optimizer.state == {}
    assert scheduler.last_epoch == 0
    for key, value in source.state_dict().items():
        assert torch.equal(value, destination.state_dict()[key])
    payload = training.checkpoint_payload(
        head=destination,
        optimizer=optimizer,
        scheduler=scheduler,
        step=restored.global_step,
        history=restored.history,
        validation={},
        configuration={},
        area_regularization_weight=0.05,
        query_entropy_weight=0.01,
        run_metadata={
            "resume_mode": restored.mode,
            "source_checkpoint": restored.source_checkpoint,
            "source_checkpoint_step": restored.source_checkpoint_step,
        },
    )
    assert payload["resume_mode"] == "weights_only"
    assert payload["source_checkpoint"] == str(path)
    assert payload["source_checkpoint_step"] == 4


def test_full_state_resume_restores_optimizer_scheduler_history_and_global_step(tmp_path: Path) -> None:
    path = tmp_path / "source.pt"
    checkpoint_with_training_state(path, step=4)
    destination = training.SatQueryGroundingHead()
    optimizer = torch.optim.AdamW(destination.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = training.make_cosine_scheduler(
        optimizer,
        total_steps=100,
        minimum_learning_rate_ratio=0.1,
    )

    restored = cli.restore_run_state(
        path,
        resume_optimizer=True,
        head=destination,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert restored.mode == "full_state"
    assert restored.global_step == 4
    assert len(restored.history) == 4
    assert restored.completed_passes == 1
    assert len(optimizer.state) == 12
    assert scheduler.last_epoch == 4
    assert list(cli.optimizer_step_numbers(restored.global_step, 7)) == [5, 6, 7]


def test_step_budget_validation_checkpoint_and_iterator_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyDataset:
        class_labels = ["vehicle"]

        def __len__(self) -> int:
            return 1

    class FakeBase(torch.nn.Module):
        def forward(self, **_inputs: object) -> SimpleNamespace:
            hidden = torch.ones(1, 4, 256)
            boxes = torch.tensor(
                [[[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]]
            )
            return SimpleNamespace(last_hidden_state=hidden, pred_boxes=boxes)

    batch = {
        "model_inputs": {"pixel_values": torch.ones(1, 1)},
        "ground_truth_boxes": torch.tensor([[0.35, 0.35, 0.65, 0.65]]),
    }
    validation_calls: list[int] = []

    def fake_validate(*_args: object, **_kwargs: object) -> dict[str, float | int]:
        validation_calls.append(len(validation_calls))
        accuracy = 0.1 + len(validation_calls) * 0.01
        return {
            "samples": 1,
            "mean_iou": accuracy,
            "median_iou": accuracy,
            "accuracy_at_025": accuracy,
            "accuracy_at_050": accuracy,
            "accuracy_at_075": accuracy,
            "average_predicted_box_area": 0.1,
            "average_ground_truth_box_area": 0.1,
            "top_1_query_usage": 1.0,
            "top_5_query_coverage": 1.0,
            "query_entropy": 0.0,
            "unique_queries_used": 1,
            "average_confidence": 0.5,
        }

    monkeypatch.setattr(cli, "load_manifest_records", lambda *_args: [{"object_class": "vehicle"}])
    monkeypatch.setattr(cli, "GroundingManifestDataset", lambda _records: TinyDataset())
    monkeypatch.setattr(cli, "load_frozen_base", lambda *_args, **_kwargs: (object(), FakeBase()))
    monkeypatch.setattr(cli, "GroundingBatchCollator", lambda _processor: None)
    monkeypatch.setattr(
        cli,
        "build_data_loader",
        lambda _dataset, *, training, **_kwargs: [batch] if training else [None],
    )
    monkeypatch.setattr(cli, "validate", fake_validate)

    args = cli.parse_args(
        [
            "--train-manifest", str(tmp_path / "train.jsonl"),
            "--validation-manifest", str(tmp_path / "validation.jsonl"),
            "--image-root", str(tmp_path),
            "--output-dir", str(tmp_path / "output"),
            "--max-steps", "7",
            "--validate-every-steps", "3",
            "--checkpoint-every-steps", "2",
            "--device", "cpu",
        ]
    )
    result = cli.run_training(args)

    assert result["global_step"] == 7
    assert result["optimizer_steps_this_run"] == 7
    assert result["validation_steps"] == [0, 3, 6, 7]
    assert result["iterator_restarts"] == 6
    assert (tmp_path / "output" / "latest.pt").is_file()
    assert (tmp_path / "output" / "best_accuracy50.pt").is_file()
    assert sorted(path.name for path in (tmp_path / "output").glob("step_*.pt")) == [
        "step_2.pt",
        "step_4.pt",
        "step_6.pt",
    ]
    for checkpoint_step in (2, 4, 6):
        periodic = torch.load(
            tmp_path / "output" / f"step_{checkpoint_step}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert periodic["step"] == checkpoint_step
        assert periodic["configuration"]["global_step"] == checkpoint_step
    latest = torch.load(tmp_path / "output" / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["step"] == 7
    assert latest["configuration"]["target_global_step"] == 7


def test_max_steps_is_scheduler_horizon_and_step_sequence_is_exact() -> None:
    horizon = cli.training_horizon_steps(max_steps=1000, epochs=30, train_batches=4535)
    steps = cli.optimizer_step_numbers(0, horizon)
    assert horizon == 1000
    assert len(steps) == 1000
    assert steps[0] == 1
    assert steps[-1] == 1000

    head = training.SatQueryGroundingHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)
    scheduler = training.make_cosine_scheduler(
        optimizer,
        total_steps=horizon,
        minimum_learning_rate_ratio=0.1,
    )
    assert scheduler.lr_lambdas[0](500) == pytest.approx(0.55)
    assert scheduler.lr_lambdas[0](1000) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--max-steps", "0"], "--max-steps must be a positive integer"),
        (["--validate-every-steps", "-1"], "--validate-every-steps must be a positive integer"),
        (["--checkpoint-every-steps", "0"], "--checkpoint-every-steps must be a positive integer"),
        (["--resume-optimizer"], "--resume-optimizer requires --resume-checkpoint"),
    ],
)
def test_invalid_step_options_are_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    required = [
        "--train-manifest", str(tmp_path / "train.jsonl"),
        "--validation-manifest", str(tmp_path / "validation.jsonl"),
        "--image-root", str(tmp_path),
        "--output-dir", str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit):
        cli.parse_args(required + arguments)
    assert message in capsys.readouterr().err


def test_early_stopping_monitors_accuracy50_with_five_validation_patience() -> None:
    stopping = training.EarlyStopping(patience=5)
    assert stopping.update(0.40) == (True, False)
    for score in (0.39, 0.40, 0.38, 0.37):
        assert stopping.update(score) == (False, False)
    assert stopping.update(0.36) == (False, True)
    assert stopping.best_accuracy50 == 0.40


def test_validation_metrics_include_geometry_and_query_usage() -> None:
    accumulator = training.ValidationAccumulator.empty()
    logits = torch.tensor([[8.0, 0.0, -1.0], [7.0, 0.0, -1.0], [0.0, 8.0, -1.0]])
    boxes = torch.tensor(
        [
            [[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]],
            [[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]],
            [[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.4, 0.4], [0.8, 0.8, 0.1, 0.1]],
        ]
    )
    targets = torch.tensor([[0.3, 0.3, 0.7, 0.7]] * 3)
    accumulator.update(logits, boxes, targets)
    metrics = accumulator.metrics()
    required = {
        "mean_iou",
        "accuracy_at_025",
        "accuracy_at_050",
        "accuracy_at_075",
        "average_predicted_box_area",
        "average_ground_truth_box_area",
        "top_1_query_usage",
        "top_5_query_coverage",
        "query_entropy",
    }
    assert required.issubset(metrics)
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["top_1_query_usage"] == pytest.approx(2 / 3)
    assert metrics["top_5_query_coverage"] == pytest.approx(1.0)


def test_training_package_has_no_production_dependencies() -> None:
    package_root = Path(training.__file__).parent
    forbidden = ("satquery_agent", "scripts.", "fastapi", "backend.py", "RemoteSensingGrounder")
    for source_path in package_root.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{source_path.name} contains forbidden dependency {token}"


def test_module_cli_runs_with_only_standalone_training_package(tmp_path: Path) -> None:
    source_package = Path(training.__file__).parent
    copied_package = tmp_path / "training"
    shutil.copytree(source_package, copied_package, ignore=shutil.ignore_patterns("__pycache__"))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "training", "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--train-manifest" in completed.stdout
    assert "--local-files-only" in completed.stdout
    assert "--max-steps" in completed.stdout
    assert "--resume-optimizer" in completed.stdout


def test_exported_pilot_step_100_checkpoint_is_backward_compatible() -> None:
    checkpoint = Path(
        os.environ.get(
            "SATQUERY_PILOT_CHECKPOINT",
            "/tmp/VRSBench/pilot_step_100.pt",
        )
    )
    if not checkpoint.is_file():
        pytest.skip("Exported pilot_step_100.pt is not available on this host")
    head = training.SatQueryGroundingHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = training.make_cosine_scheduler(
        optimizer,
        total_steps=1000,
        minimum_learning_rate_ratio=0.1,
    )
    restored = cli.restore_run_state(
        checkpoint,
        resume_optimizer=True,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert restored.global_step == 100
    assert restored.mode == "full_state"
    assert len(head.state_dict()) == 12
    assert len(optimizer.state) == 12
    assert scheduler.last_epoch == 100
