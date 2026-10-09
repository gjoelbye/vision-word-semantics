"""The checkpoint format: written by training, read back on resume and for analysis.

A checkpoint holds the backbone under an `image_backbone.` prefix beside the
contrastive projection and temperature. Loading a finetuned backbone into its
pretrained architecture drops the last two. A saved key the model does not
have is an error, because its weights would otherwise be lost silently.
"""
from __future__ import annotations

from pathlib import Path

import torch

# Weights belonging to the image backbone. Everything without this prefix
# (`image_proj.weight`, `logit_scale`) is part of the contrastive head.
_PREFIX = "image_backbone."

# The contrastive head: the projection into the tower's space and the learned
# temperature. Both are discarded, and a checkpoint with any other head is refused.
_HEAD = {"image_proj.weight", "logit_scale"}


def backbone_state(model) -> dict:
    """The weights of a `JointModel` under their checkpoint keys."""
    state = {f"{_PREFIX}{k}": v.detach().cpu()
             for k, v in model.backbone.state_dict().items()}
    state["image_proj.weight"] = model.image_proj.weight.detach().cpu()
    state["logit_scale"] = model.logit_scale_raw.detach().cpu()
    return state


def _backbone_keys(state: dict) -> dict:
    """Checkpoint keys back to `JointModel` keys, the inverse of `backbone_state`."""
    out = {}
    for key, value in state.items():
        if key.startswith(_PREFIX):
            out[f"backbone.{key[len(_PREFIX):]}"] = value
        elif key == "image_proj.weight":
            out["image_proj.weight"] = value
        elif key == "logit_scale":
            out["logit_scale_raw"] = value
    return out


def _resolve(key: str, expected: set[str]) -> str:
    """Match one saved key against the live model's keys, as saved or with a
    leading `model.` added or removed.

    Transformers versions differ in whether the DINOv3 block stack sits under a
    `model.` submodule. A key that matches none of the three is returned
    unchanged and surfaces as an unexpected key.
    """
    if key in expected:
        return key
    if f"model.{key}" in expected:
        return f"model.{key}"
    if key.startswith("model.") and key[len("model."):] in expected:
        return key[len("model."):]
    return key


def load_backbone_state(ckpt_path: str | Path, model, *, expect_repo: str,
                        expect_step: int) -> dict:
    """Load the image backbone from `ckpt_path` into `model`, in place.

    Returns the step the weights were saved at and the tensor count. Raises if
    the checkpoint's backbone repository or step differs from `expect_repo` or
    `expect_step`.
    """
    # Memory-mapped, so each tensor is read once as it is copied into the model.
    ckpt_path = Path(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True,
                         mmap=True)
    if "model" not in payload:
        raise RuntimeError(f"{ckpt_path} has no 'model' entry")

    cfg = payload.get("cfg", {})
    if cfg.get("backbone_repo") != expect_repo:
        raise RuntimeError(
            f"{ckpt_path} was trained on {cfg.get('backbone_repo')!r}, but the "
            f"registry pairs it with {expect_repo!r}")
    if int(payload["step"]) != expect_step:
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

    return {"train_step": int(payload["step"]), "n_tensors": len(state)}
