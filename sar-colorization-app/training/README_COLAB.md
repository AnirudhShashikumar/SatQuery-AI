# Grounding Specialist v1.1 — standalone Colab trainer

The `training/` directory is self-contained. It does not import SatQuery,
FastAPI, backend services, inference specialists, or evaluation scripts.

## Colab setup

Copy the complete `training/` directory into the Colab working directory, then
install its small dependency set:

```bash
pip install -r training/requirements-colab.txt
```

Start training from the directory that contains `training/`:

```bash
python -m training \
  --train-manifest /content/data/train.jsonl \
  --validation-manifest /content/data/validation.jsonl \
  --image-root /content/data/images \
  --output-dir /content/checkpoints/grounding-specialist-v1.1 \
  --epochs 30 \
  --batch-size 8
```

Grounding DINO is downloaded through Transformers on the first run. Use
`--cache-dir PATH` to select the Hugging Face cache or `--local-files-only` for
an offline runtime whose cache is already populated.

The manifest is JSON Lines and each record must contain `object_class`,
`referring_sentence`, `ground_truth_bbox` (normalized xyxy), and either `image`
or `local_image_path`.

Outputs include `latest.pt`, `best_accuracy50.pt`, and `training_log.jsonl`.
The checkpoints retain the v1 format and add the v1.1 recipe metadata.

## Controlled 1,000-step experiment

This command loads the pilot specialist weights strictly, starts a fresh
optimizer and cosine scheduler at global step 0, runs exactly 1,000 optimizer
steps, validates at steps 0, 100, 200, ..., 1,000, and writes periodic
checkpoints at every 100-step boundary:

```bash
python -m training \
  --train-manifest /content/data/train.jsonl \
  --validation-manifest /content/data/validation.jsonl \
  --image-root /content/data/images \
  --output-dir /content/checkpoints/grounding-specialist-v1.1-step1000 \
  --resume-checkpoint /content/pilot_step_100.pt \
  --max-steps 1000 \
  --validate-every-steps 100 \
  --checkpoint-every-steps 100 \
  --batch-size 8
```

`--max-steps` is an absolute global-step target and overrides the epoch-derived
horizon. In a fixed-step experiment it is authoritative: early-stopping events
are still tracked, but they do not truncate the requested optimizer-step budget.

By default a resume checkpoint supplies only specialist weights. History,
optimizer, scheduler, and global step are reset, and the checkpoint path and
source step are recorded in new checkpoint metadata. Add `--resume-optimizer`
for a full-state continuation; for example, a step-100 pilot resumed with
`--max-steps 1000` performs 900 additional updates and finishes at global step
1,000.

When `--validate-every-steps` is omitted, step-limited runs validate initially
and at the final step, while epoch-based runs retain epoch-boundary validation.
When `--checkpoint-every-steps` is omitted, `latest.pt` is still refreshed at
every validation and at completion.
