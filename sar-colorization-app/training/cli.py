"""Standalone command-line trainer for Grounding Specialist v1.1."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterator, Mapping, Optional, Sequence, TypeVar

import numpy as np
import torch

from .grounding_specialist_v11 import (
    DEFAULT_AREA_REGULARIZATION_WEIGHT,
    DEFAULT_QUERY_ENTROPY_WEIGHT,
    EarlyStopping,
    GroundingBatchCollator,
    GroundingManifestDataset,
    LoadedCheckpoint,
    SatQueryGroundingHead,
    build_data_loader,
    checkpoint_payload,
    grounding_specialist_losses,
    load_frozen_base,
    load_manifest_records,
    load_training_checkpoint,
    make_cosine_scheduler,
    move_model_inputs,
    save_checkpoint,
    validate,
)


LOGGER = logging.getLogger("grounding_specialist.training")
Batch = TypeVar("Batch")


@dataclass(frozen=True)
class ResumeState:
    mode: str
    global_step: int
    history: list[dict[str, Any]]
    validation_history: list[dict[str, Any]]
    completed_passes: int
    source_checkpoint: Optional[str]
    source_checkpoint_step: Optional[int]
    loaded: Optional[LoadedCheckpoint]


class RestartableDataIterator(Generic[Batch]):
    """Restart a finite data loader cleanly whenever it reaches StopIteration."""

    def __init__(self, loader: Any, *, completed_passes: int = 0) -> None:
        if len(loader) <= 0:
            raise ValueError("Training loader must contain at least one batch")
        self.loader = loader
        self.completed_passes = int(completed_passes)
        self.restart_count = 0
        self._iterator: Iterator[Batch] = iter(loader)

    def next_batch(self) -> Batch:
        try:
            return next(self._iterator)
        except StopIteration:
            self.completed_passes += 1
            self.restart_count += 1
            self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration as error:
                raise ValueError("Training loader became empty after iterator restart") from error


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Restore optimizer, scheduler, history, and global step in addition to specialist weights.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Absolute optimizer-step target. Overrides the epoch-derived training horizon.",
    )
    parser.add_argument(
        "--validate-every-steps",
        type=int,
        help="Run validation at every positive multiple of this global-step interval.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        help="Save latest.pt and step_<global_step>.pt at this global-step interval.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--area-regularization-weight",
        type=float,
        default=DEFAULT_AREA_REGULARIZATION_WEIGHT,
    )
    parser.add_argument("--query-entropy-weight", type=float, default=DEFAULT_QUERY_ENTROPY_WEIGHT)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable Hugging Face downloads and use only an existing local model cache.",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    for name in ("max_steps", "validate_every_steps", "checkpoint_every_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if args.resume_optimizer and args.resume_checkpoint is None:
        parser.error("--resume-optimizer requires --resume-checkpoint")
    if args.batch_size <= 0 or args.validation_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.early_stopping_patience <= 0:
        parser.error("--early-stopping-patience must be positive")
    if args.validation_limit is not None and args.validation_limit <= 0:
        parser.error("--validation-limit must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.minimum_learning_rate < 0 or args.learning_rate <= 0:
        parser.error("learning rates must be non-negative and --learning-rate must be positive")
    if args.minimum_learning_rate > args.learning_rate:
        parser.error("--minimum-learning-rate cannot exceed --learning-rate")
    if args.area_regularization_weight < 0 or args.query_entropy_weight < 0:
        parser.error("regularization weights must be non-negative")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def training_horizon_steps(*, max_steps: Optional[int], epochs: int, train_batches: int) -> int:
    if train_batches <= 0:
        raise ValueError("train_batches must be positive")
    return int(max_steps) if max_steps is not None else int(epochs) * int(train_batches)


def validation_is_due(
    global_step: int,
    *,
    validate_every_steps: Optional[int],
    train_batches: int,
    step_limited: bool,
) -> bool:
    if global_step <= 0:
        return global_step == 0
    if validate_every_steps is not None:
        return global_step % validate_every_steps == 0
    return not step_limited and global_step % train_batches == 0


def checkpoint_is_due(global_step: int, checkpoint_every_steps: Optional[int]) -> bool:
    return (
        global_step > 0
        and checkpoint_every_steps is not None
        and global_step % checkpoint_every_steps == 0
    )


def optimizer_step_numbers(starting_global_step: int, target_global_step: int) -> range:
    if starting_global_step < 0:
        raise ValueError("starting_global_step cannot be negative")
    if target_global_step < starting_global_step:
        raise ValueError("target_global_step cannot precede starting_global_step")
    return range(starting_global_step + 1, target_global_step + 1)


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


def restore_run_state(
    checkpoint: Optional[Path],
    *,
    resume_optimizer: bool,
    head: SatQueryGroundingHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> ResumeState:
    if checkpoint is None:
        return ResumeState("fresh", 0, [], [], 0, None, None, None)
    if resume_optimizer:
        loaded = load_training_checkpoint(
            checkpoint,
            head,
            optimizer=optimizer,
            scheduler=scheduler,
            require_training_state=True,
        )
        if scheduler.last_epoch != loaded.step:
            raise ValueError(
                "Checkpoint scheduler state is incompatible with its global step: "
                f"scheduler={scheduler.last_epoch}, checkpoint={loaded.step}"
            )
        return ResumeState(
            mode="full_state",
            global_step=loaded.step,
            history=loaded.history,
            validation_history=list(loaded.configuration.get("validation_history", [])),
            completed_passes=int(loaded.configuration.get("completed_passes", 0)),
            source_checkpoint=str(checkpoint),
            source_checkpoint_step=loaded.step,
            loaded=loaded,
        )
    loaded = load_training_checkpoint(checkpoint, head)
    return ResumeState(
        mode="weights_only",
        global_step=0,
        history=[],
        validation_history=[],
        completed_passes=0,
        source_checkpoint=str(checkpoint),
        source_checkpoint_step=loaded.step,
        loaded=loaded,
    )


def _checkpoint_configuration(
    args: argparse.Namespace,
    *,
    train_samples: int,
    validation_samples: int,
    train_batches: int,
    target_global_step: int,
    global_step: int,
    completed_passes: int,
    class_counts: Counter[str],
    early_stopping: EarlyStopping,
    validation_history: list[dict[str, Any]],
    resume_state: ResumeState,
) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "completed_epoch": completed_passes,
        "completed_passes": completed_passes,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "max_gradient_norm": args.max_gradient_norm,
        "validation_limit": args.validation_limit,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "train_batches": train_batches,
        "class_counts": dict(sorted(class_counts.items())),
        "max_steps": args.max_steps,
        "target_global_step": target_global_step,
        "global_step": global_step,
        "validate_every_steps": args.validate_every_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_best_accuracy50": early_stopping.best_accuracy50,
        "validations_without_improvement": early_stopping.validations_without_improvement,
        "validation_history": validation_history,
        "seed": args.seed,
        "local_files_only": args.local_files_only,
        "resume_mode": resume_state.mode,
        "starting_global_step": resume_state.global_step,
        "source_checkpoint": resume_state.source_checkpoint,
        "source_checkpoint_step": resume_state.source_checkpoint_step,
    }


def _run_metadata(resume_state: ResumeState) -> dict[str, Any]:
    return {
        "resume_mode": resume_state.mode,
        "source_checkpoint": resume_state.source_checkpoint,
        "source_checkpoint_step": resume_state.source_checkpoint_step,
        "starting_global_step": resume_state.global_step,
    }


def _save_training_state(
    path: Path,
    *,
    head: SatQueryGroundingHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    history: list[dict[str, Any]],
    validation_metrics: Mapping[str, Any],
    configuration: Mapping[str, Any],
    args: argparse.Namespace,
    resume_state: ResumeState,
) -> None:
    payload = checkpoint_payload(
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        history=history,
        validation=validation_metrics,
        configuration=configuration,
        area_regularization_weight=args.area_regularization_weight,
        query_entropy_weight=args.query_entropy_weight,
        run_metadata=_run_metadata(resume_state),
    )
    save_checkpoint(path, payload)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    sampler_generator = seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.output_dir / "training_log.jsonl"

    validation_root = args.validation_image_root or args.image_root
    train_records = load_manifest_records(args.train_manifest, args.image_root)
    validation_records = load_manifest_records(args.validation_manifest, validation_root)
    if args.validation_limit is not None:
        validation_records = validation_records[: args.validation_limit]
    train_dataset = GroundingManifestDataset(train_records)
    validation_dataset = GroundingManifestDataset(validation_records)
    class_counts = Counter(train_dataset.class_labels)

    processor, base_model = load_frozen_base(
        device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )
    collator = GroundingBatchCollator(processor)
    train_loader = build_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        training=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=sampler_generator,
    )
    validation_loader = build_data_loader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        training=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    train_batches = len(train_loader)
    target_global_step = training_horizon_steps(
        max_steps=args.max_steps,
        epochs=args.epochs,
        train_batches=train_batches,
    )

    head = SatQueryGroundingHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_cosine_scheduler(
        optimizer,
        total_steps=target_global_step,
        minimum_learning_rate_ratio=args.minimum_learning_rate / args.learning_rate,
    )
    resume_state = restore_run_state(
        args.resume_checkpoint,
        resume_optimizer=args.resume_optimizer,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if resume_state.global_step >= target_global_step:
        raise ValueError(
            f"Starting global step {resume_state.global_step} must be below training target "
            f"{target_global_step}. Increase --max-steps or --epochs."
        )

    history = list(resume_state.history)
    validation_history = list(resume_state.validation_history)
    global_step = resume_state.global_step
    data_iterator = RestartableDataIterator(
        train_loader,
        completed_passes=resume_state.completed_passes,
    )
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    if resume_state.mode == "full_state" and resume_state.loaded is not None:
        configuration = resume_state.loaded.configuration
        early_stopping.best_accuracy50 = float(
            configuration.get(
                "early_stopping_best_accuracy50",
                resume_state.loaded.validation.get("accuracy_at_050", -float("inf")),
            )
        )
        early_stopping.validations_without_improvement = int(
            configuration.get("validations_without_improvement", 0)
        )

    trainable_head_parameters = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    trainable_base_parameters = sum(
        parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad
    )
    startup = {
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "train_batches": train_batches,
        "max_steps": args.max_steps,
        "target_global_step": target_global_step,
        "validate_every_steps": args.validate_every_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "resume_mode": resume_state.mode,
        "starting_global_step": global_step,
        "balanced_sampler": True,
        "trainable_head_parameters": trainable_head_parameters,
        "trainable_base_parameters": trainable_base_parameters,
    }
    LOGGER.info(
        "Startup: train_records=%d validation_records=%d train_batches=%d max_steps=%s "
        "validation_interval=%s checkpoint_interval=%s resume_mode=%s starting_global_step=%d "
        "balanced_sampler=%s trainable_head_parameters=%d trainable_base_parameters=%d",
        startup["train_records"],
        startup["validation_records"],
        startup["train_batches"],
        startup["max_steps"],
        startup["validate_every_steps"],
        startup["checkpoint_every_steps"],
        startup["resume_mode"],
        startup["starting_global_step"],
        startup["balanced_sampler"],
        startup["trainable_head_parameters"],
        startup["trainable_base_parameters"],
    )
    append_event(event_log, {"event": "startup", **startup})

    latest_validation: dict[str, Any] = {}
    last_validated_step: Optional[int] = None
    stopped_early = False
    early_stopping_triggered = False
    training_started = time.perf_counter()

    def configuration() -> dict[str, Any]:
        return _checkpoint_configuration(
            args,
            train_samples=len(train_dataset),
            validation_samples=len(validation_dataset),
            train_batches=train_batches,
            target_global_step=target_global_step,
            global_step=global_step,
            completed_passes=data_iterator.completed_passes,
            class_counts=class_counts,
            early_stopping=early_stopping,
            validation_history=validation_history,
            resume_state=resume_state,
        )

    def save_named(path: Path) -> None:
        _save_training_state(
            path,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            history=history,
            validation_metrics=latest_validation,
            configuration=configuration(),
            args=args,
            resume_state=resume_state,
        )

    def run_validation_event(reason: str) -> bool:
        nonlocal latest_validation, last_validated_step, early_stopping_triggered
        started = time.perf_counter()
        latest_validation = dict(validate(base_model, head, validation_loader, device))
        validation_event = {
            "event": "validation",
            "validation_index": len(validation_history),
            "reason": reason,
            "global_step": global_step,
            "step": global_step,
            "seconds": time.perf_counter() - started,
            **latest_validation,
        }
        validation_history.append(validation_event)
        append_event(event_log, validation_event)
        improved, should_stop = early_stopping.update(float(latest_validation["accuracy_at_050"]))
        last_validated_step = global_step
        save_named(args.output_dir / "latest.pt")
        best_path = args.output_dir / "best_accuracy50.pt"
        if improved or not best_path.exists():
            save_named(best_path)
        LOGGER.info(
            "validation step=%d reason=%s mean_iou=%.4f acc25=%.4f acc50=%.4f acc75=%.4f "
            "pred_area=%.4f gt_area=%.4f top1=%.4f top5=%.4f entropy=%.4f%s",
            global_step,
            reason,
            latest_validation["mean_iou"],
            latest_validation["accuracy_at_025"],
            latest_validation["accuracy_at_050"],
            latest_validation["accuracy_at_075"],
            latest_validation["average_predicted_box_area"],
            latest_validation["average_ground_truth_box_area"],
            latest_validation["top_1_query_usage"],
            latest_validation["top_5_query_coverage"],
            latest_validation["query_entropy"],
            " [best]" if improved else "",
        )
        early_stopping_triggered = early_stopping_triggered or should_stop
        return should_stop

    # Every run validates before its first optimizer update. Fresh and
    # weights-only runs are at step 0; full-state resumes validate at the
    # restored authoritative global step.
    run_validation_event("initial")

    for next_global_step in optimizer_step_numbers(global_step, target_global_step):
        batch = data_iterator.next_batch()
        head.train()
        inputs = move_model_inputs(batch["model_inputs"], device)
        targets = batch["ground_truth_boxes"].to(device)
        with torch.no_grad():
            base_outputs = base_model(**inputs)
            hidden_state = base_outputs.last_hidden_state.detach()
            base_boxes = base_outputs.pred_boxes.detach()
        query_logits, refined_boxes = head(hidden_state, base_boxes)
        losses = grounding_specialist_losses(
            query_logits,
            refined_boxes,
            base_boxes,
            targets,
            area_regularization_weight=args.area_regularization_weight,
            query_entropy_weight=args.query_entropy_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_gradient_norm)
        optimizer.step()
        scheduler.step()
        global_step = next_global_step
        row = {
            "step": global_step,
            "global_step": global_step,
            "epoch": data_iterator.completed_passes + 1,
            "loss": float(losses["loss"].detach().cpu()),
            "classification_loss": float(losses["classification_loss"].detach().cpu()),
            "l1_loss": float(losses["l1_loss"].detach().cpu()),
            "giou_loss": float(losses["giou_loss"].detach().cpu()),
            "area_loss": float(losses["area_loss"].detach().cpu()),
            "diversity_loss": float(losses["diversity_loss"].detach().cpu()),
            "query_entropy": float(losses["query_entropy"].detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        append_event(event_log, {"event": "training_step", **row})
        if global_step == 1 or global_step % args.log_every == 0 or global_step == target_global_step:
            LOGGER.info(
                "step=%d/%d epoch_pass=%d loss=%.4f area=%.4f diversity=%.4f",
                global_step,
                target_global_step,
                data_iterator.completed_passes + 1,
                row["loss"],
                row["area_loss"],
                row["diversity_loss"],
            )

        validated = False
        interval_validation_due = validation_is_due(
            global_step,
            validate_every_steps=args.validate_every_steps,
            train_batches=train_batches,
            step_limited=args.max_steps is not None,
        )
        final_step_due = global_step == target_global_step
        if interval_validation_due or final_step_due:
            reason = "interval" if interval_validation_due else "final"
            should_stop = run_validation_event(reason)
            validated = True
            # An explicit max-step run is a controlled fixed-budget experiment;
            # keep counting early-stopping events but do not truncate its budget.
            if should_stop and args.max_steps is None:
                stopped_early = True

        if checkpoint_is_due(global_step, args.checkpoint_every_steps):
            if not validated:
                save_named(args.output_dir / "latest.pt")
            save_named(args.output_dir / f"step_{global_step}.pt")
            append_event(
                event_log,
                {"event": "periodic_checkpoint", "global_step": global_step, "step": global_step},
            )
        if stopped_early:
            LOGGER.info(
                "Early stopping after %d validation events without Accuracy@0.50 improvement",
                args.early_stopping_patience,
            )
            break

    if last_validated_step != global_step:
        run_validation_event("final")
    else:
        save_named(args.output_dir / "latest.pt")

    result = {
        "step": global_step,
        "global_step": global_step,
        "target_global_step": target_global_step,
        "optimizer_steps_this_run": global_step - resume_state.global_step,
        "completed_passes": data_iterator.completed_passes,
        "iterator_restarts": data_iterator.restart_count,
        "stopped_early": stopped_early,
        "early_stopping_triggered": early_stopping_triggered,
        "best_accuracy50": early_stopping.best_accuracy50,
        "validation_steps": [int(row["global_step"]) for row in validation_history],
        "latest_validation": latest_validation,
        "runtime_seconds": time.perf_counter() - training_started,
        "resume_mode": resume_state.mode,
        "latest_checkpoint": str(args.output_dir / "latest.pt"),
        "best_checkpoint": str(args.output_dir / "best_accuracy50.pt"),
    }
    append_event(event_log, {"event": "training_complete", **result})
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_training(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
