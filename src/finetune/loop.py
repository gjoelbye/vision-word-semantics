"""The training loop, with resume from a saved state.

``latest.pt`` holds the weights, optimizer moments and random generator states
at a fixed cadence, so an interrupted run continues the same trajectory.
``best.slim.pt`` holds the weights at the lowest validation loss and is the
file :func:`src.models.load` reads for a finetuned model. Both are written
through a temporary file and renamed. The learning rate warms up linearly and
then decays on a cosine.
"""
from __future__ import annotations

import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .objective import cached_step


def learning_rate_scale(step: int, warmup: int, total: int) -> float:
    """Linear to one over ``warmup``, then cosine to zero at ``total``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model, recipe):
    return torch.optim.AdamW(model.parameter_groups(), lr=recipe.lr_backbone,
                             betas=(0.9, 0.98), eps=1e-6,
                             weight_decay=recipe.weight_decay)


def rng_state() -> dict:
    return {"python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state["cuda"]])


def save(payload: dict, path: Path) -> None:
    """Write through a temporary file, then rename, so a killed run leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _payload(model, optimizer, recipe, step: int, best_loss: float,
             best_step: int, *, slim: bool) -> dict:
    """The checkpoint: weights and the run record, plus optimizer and
    generator states unless ``slim``."""
    payload = {"model": model.backbone_state(), "step": int(step),
               "best_step": int(best_step), "best_val_loss": float(best_loss),
               "cfg": recipe.to_dict() | {"backbone_repo": model.spec.repo,
                                          "total_steps": recipe.steps}}
    if not slim:
        payload |= {"optimizer": optimizer.state_dict(), "rng": rng_state()}
    return payload


@torch.no_grad()
def validate(model, loader, *, batches: int, device: str,
             autocast_dtype) -> float:
    """Mean InfoNCE loss over the first ``batches`` validation micro-batches,
    each scored against its own negatives only."""
    from .objective import infonce

    model.eval()
    total, seen = 0.0, 0
    for images, captions in loader:
        if seen >= batches:
            break
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        image = model.encode_image(images).float()
        text = model.encode_text(captions).float()
        total += float(infonce(image, text, model.logit_scale()))
        seen += 1
    model.train()
    return total / max(1, seen)


def run(model, recipe, train_loader, val_loader, *, out: Path,
        device: str = "cuda", autocast_dtype=torch.bfloat16,
        checkpoint_every: int = 500, validate_every: int = 500,
        validate_batches: int = 32, patience: int = 0) -> tuple[Path, str]:
    """Train ``model`` and return the path of ``best.slim.pt`` and why the run ended.

    The run resumes from ``latest.pt`` in ``out`` when one exists, and SIGTERM
    or SIGINT saves it and returns an outcome starting with "interrupted". The
    run stops early once ``patience`` validations in a row fail to improve on
    the best; zero runs the whole schedule.
    """
    out.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, recipe)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    start, best_loss, best_step = 0, float("inf"), 0
    stale = 0

    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(_backbone_keys(state["model"]), strict=False)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start, best_loss = int(state["step"]), float(state["best_val_loss"])
        best_step = int(state["best_step"])
        del state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"resuming at step {start:,} of {recipe.steps:,}", flush=True)
    else:
        torch.manual_seed(recipe.seed)
        np.random.seed(recipe.seed)
        random.seed(recipe.seed)

    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))

    batches = iter(train_loader)
    model.train()
    t0 = time.time()
    for step in range(start, recipe.steps):
        scale = learning_rate_scale(step, recipe.warmup, recipe.steps)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        micro = []
        for _ in range(recipe.accumulation):
            try:
                images, captions = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                images, captions = next(batches)
            micro.append((images.to(device, non_blocking=True),
                          captions.to(device, non_blocking=True)))

        optimizer.zero_grad(set_to_none=True)
        result = cached_step(model, micro, autocast_dtype=autocast_dtype)
        if recipe.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
        optimizer.step()
        model.clamp_logit_scale()

        done = step + 1
        if done % 50 == 0 or step == start:
            rate = (done - start) / max(1e-9, time.time() - t0)
            print(f"  step {done:>6,}/{recipe.steps:,}  loss {result.loss:.4f}  "
                  f"acc {result.image_accuracy:.3f}  lr {scale:.3f}  "
                  f"{rate:.2f} step/s", flush=True)

        if validate_every and done % validate_every == 0:
            loss = validate(model, val_loader, batches=validate_batches,
                            device=device, autocast_dtype=autocast_dtype)
            print(f"  step {done:>6,} validation {loss:.4f}"
                  f"{'  (best)' if loss < best_loss else ''}", flush=True)
            if loss < best_loss:
                best_loss, best_step = loss, done
                stale = 0
                save(_payload(model, optimizer, recipe, done, best_loss,
                              best_step, slim=True), out / "best.slim.pt")
            else:
                stale += 1
                if patience and stale >= patience:
                    save(_payload(model, optimizer, recipe, done, best_loss,
                                  best_step, slim=False), latest)
                    print(f"  early stop at step {done:,}: {stale} validations "
                          f"without improving on {best_loss:.4f} at step "
                          f"{best_step:,}", flush=True)
                    return out / "best.slim.pt", (
                        f"early stop at step {done} after {stale} validations "
                        f"without improvement")

        if checkpoint_every and done % checkpoint_every == 0 or stop[0]:
            save(_payload(model, optimizer, recipe, done, best_loss, best_step,
                          slim=False), latest)
        if stop[0]:
            print("stopped; latest.pt written")
            return out / "best.slim.pt", f"interrupted at step {done}"

    if best_step == 0:
        # No validation recorded a best, so the final weights are the ones kept.
        save(_payload(model, optimizer, recipe, recipe.steps, best_loss,
                      recipe.steps, slim=True), out / "best.slim.pt")
    save(_payload(model, optimizer, recipe, recipe.steps, best_loss, best_step,
                  slim=False), latest)
    return out / "best.slim.pt", "completed the schedule"


def _backbone_keys(state: dict) -> dict:
    """Checkpoint keys back to module keys, the inverse of ``backbone_state``."""
    out = {}
    for key, value in state.items():
        if key.startswith("image_backbone."):
            out[f"backbone.{key[len('image_backbone.'):]}"] = value
        elif key == "image_proj.weight":
            out["image_proj.weight"] = value
        elif key == "logit_scale":
            out["logit_scale_raw"] = value
    return out
