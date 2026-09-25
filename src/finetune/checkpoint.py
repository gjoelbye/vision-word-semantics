"""Load a finetuned backbone's weights into its pretrained architecture.

A checkpoint holds the backbone under an ``image_backbone.`` prefix beside the
contrastive projection and temperature, which are dropped. A saved key the
model does not have is an error, because its weights would otherwise be lost
silently.
"""
from __future__ import annotations

from pathlib import Path

import torch

#: Weights belonging to the image backbone. Everything without this prefix
#: (``image_proj.weight``, ``logit_scale``) is part of the contrastive head.
_PREFIX = "image_backbone."


def _resolve(key: str, expected: set[str]) -> str:
    """Match one saved key against the live model's keys, as saved or with a
    leading ``model.`` added or removed.

    Transformers versions differ in whether the DINOv3 block stack sits under a
    ``model.`` submodule. A key that matches none of the three is returned
    unchanged and surfaces as an unexpected key.
    """
    if key in expected:
        return key
    if f"model.{key}" in expected:
        return f"model.{key}"
    if key.startswith("model.") and key[len("model."):] in expected:
        return key[len("model."):]
    return key


#: The contrastive head: the projection into the tower's space and the learned
#: temperature. Both are discarded, and a checkpoint with any other head is refused.
_HEAD = {"image_proj.weight", "logit_scale"}


def load_backbone_state(ckpt_path: str | Path, model, *,
                        expect_repo: str | None = None,
                        expect_step: int | None = None) -> dict:
    """Load the image backbone from ``ckpt_path`` into ``model``, in place.

    Returns the checkpoint's record: the step the weights were saved at, the
    step and value of the best validation loss, and the tensor count. Raises if
    the checkpoint's backbone repository or step differs from ``expect_repo`` or
    ``expect_step``.
    """
    # Memory-mapped, so each tensor is read once as it is copied into the model.
    ckpt_path = Path(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False,
                         mmap=True)
    if "model" not in payload:
        raise RuntimeError(f"{ckpt_path} has no 'model' entry")

    cfg = payload.get("cfg", {})
    if expect_repo is not None and cfg.get("backbone_repo") != expect_repo:
        raise RuntimeError(
            f"{ckpt_path} was trained on {cfg.get('backbone_repo')!r}, but the "
            f"registry pairs it with {expect_repo!r}")
    if expect_step is not None and int(payload["step"]) != expect_step:
        raise RuntimeError(
            f"{ckpt_path} records step {payload['step']}, the registry says "
            f"{expect_step}")

    expected = set(model.state_dict().keys())
    state = {_resolve(k[len(_PREFIX):], expected): v
             for k, v in payload["model"].items() if k.startswith(_PREFIX)}
    if not state:
        raise RuntimeError(f"{ckpt_path} has no {_PREFIX}* keys")
    dropped = {k for k in payload["model"] if not k.startswith(_PREFIX)}
    if dropped != _HEAD:
        raise RuntimeError(
            f"{ckpt_path} drops {sorted(dropped)}, expected {sorted(_HEAD)}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"{ckpt_path.parent.name}: {len(unexpected)} keys the model does "
            f"not expect, so their weights were discarded: {sorted(unexpected)[:5]}")
    if missing:
        print(f"  {len(missing)} keys kept at pretrained values: "
              f"{sorted(missing)[:5]}")

    return {"train_step": int(payload["step"]),
            "best_step": int(payload["best_step"]),
            "best_val_loss": float(payload["best_val_loss"]),
            "n_tensors": len(state)}
