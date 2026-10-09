"""Finetune a backbone contrastively on SPEECH-COCO against a frozen caption tower.

`--backbone B --tower T` runs one finetune; `--all` runs the twenty, smallest
backbone first, each in its own process, skipping any whose provenance exists.
A run resumes from `latest.pt` when one exists. It writes
`data/checkpoints/<key>/best.slim.pt` (the weights at the lowest validation loss)
and `provenance.json`, then deletes `latest.pt`. The schedule follows from
the corpus size: at most three epochs and 10,000 steps, warmup a tenth of the
run, validation every twelfth and a checkpoint every sixth (each between 10 and
500 steps), and an early stop after three validations without improvement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys

import torch

from src.config import CHECKPOINTS, ensure_dirs
from src.finetune import data
from src.finetune.model import JointModel
from src.finetune.training import (MAX_EPOCHS, MICRO_BATCH, PATIENCE, Recipe,
                                   run, schedule)
from src.models import FINETUNES, MODELS, PARAMS_M

WORKERS = 8


def sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def train(backbone: str, tower: str) -> None:
    """Run one finetune and write its best checkpoint and provenance."""
    kept = data.kept_size()
    plan = schedule(kept)
    recipe = Recipe(backbone=backbone, tower=tower, steps=plan["steps"],
                    warmup=plan["warmup"], micro_batch=MICRO_BATCH[backbone])
    out = CHECKPOINTS / recipe.name
    print(f"{recipe.name} | {kept:,} samples after the text filter | "
          f"{recipe.steps:,} steps, warmup {recipe.warmup}, validation every "
          f"{plan['validate_every']}, checkpoint every {plan['checkpoint_every']} | "
          f"{recipe.batch} images per step ({recipe.accumulation} x {recipe.micro_batch})")

    cache = data.CaptionCache(tower)
    exclude = data.excluded_keys()
    print(f"caption cache: {len(cache):,} vectors | text filter: "
          f"{len(exclude):,} samples excluded")

    model = JointModel(recipe)
    train_loader = data.pipeline(recipe, train=True, cache=cache,
                                 exclude=exclude, workers=WORKERS)
    val_loader = data.pipeline(recipe, train=False, cache=cache,
                               exclude=exclude, workers=WORKERS)

    started = dt.datetime.now().isoformat(timespec="seconds")
    path, outcome = run(model, recipe, train_loader, val_loader, out=out,
                        checkpoint_every=plan["checkpoint_every"],
                        validate_every=plan["validate_every"], patience=PATIENCE)
    if outcome.startswith("interrupted"):
        raise SystemExit(f"{recipe.name}: {outcome}; run again to resume")

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    (out / "provenance.json").write_text(json.dumps({
        "model": recipe.name,
        "base_model": backbone,
        "base_repo": MODELS[backbone].repo,
        "corpus": data.DESCRIBED,
        "text_tower": tower,
        "train_step": int(payload["step"]),
        "best_step": int(payload["best_step"]),
        "best_val_loss": float(payload["best_val_loss"]),
        "total_steps": recipe.steps,
        "outcome": outcome,
        "max_epochs": MAX_EPOCHS,
        "epochs_seen": round(recipe.steps * recipe.batch / kept, 3),
        "corpus_kept": kept,
        "patience": PATIENCE,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "recipe": recipe.to_dict(),
        "shard_fingerprint": data.fingerprint(),
        "n_excluded": len(exclude),
        "caption_cache": True,
        "source": f"scripts/finetune/train.py --backbone {backbone} --tower {tower}",
        "started": started,
        "recorded": dt.date.today().isoformat(),
    }, indent=2) + "\n")
    (out / "latest.pt").unlink(missing_ok=True)
    print(f"wrote checkpoints/{recipe.name}/best.slim.pt and its provenance")


def train_all() -> None:
    """Run every finetune that has no provenance, smallest backbone first, each in its own process."""
    failed = []
    for key in sorted(FINETUNES, key=lambda k: PARAMS_M[k]):
        if (CHECKPOINTS / key / "provenance.json").exists():
            print(f"{key}: finished, skipping")
            continue
        spec = MODELS[key]
        done = subprocess.run([sys.executable, __file__,
                               "--backbone", spec.extra["base"],
                               "--tower", spec.extra["tower"]])
        if done.returncode:
            failed.append(key)
    if failed:
        raise SystemExit(f"did not finish: {', '.join(failed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=sorted(MICRO_BATCH))
    ap.add_argument("--tower", choices=sorted({MODELS[k].extra["tower"]
                                               for k in FINETUNES}))
    ap.add_argument("--all", action="store_true",
                    help="run every finetune that has no provenance yet")
    a = ap.parse_args()
    ensure_dirs()
    if a.all:
        train_all()
    elif a.backbone and a.tower:
        train(a.backbone, a.tower)
    else:
        ap.error("pass --backbone and --tower, or --all")


if __name__ == "__main__":
    main()
