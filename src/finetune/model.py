"""The image backbone joined to a cached caption tower.

The backbone is built by :func:`src.models.load`, the call that embeds the
pretrained model, so the finetune differs from its twin in weights alone. Two
parameters join the two sides: a projection into the tower's space and a
learned temperature. Both are dropped when the checkpoint is read back.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import has_class_token, hidden_size, load
from .recipe import Recipe

#: Starting temperature as a log scale, CLIP's 1/0.07; the value is learned.
INITIAL_LOGIT_SCALE = math.log(1 / 0.07)

#: Ceiling on the learned log scale, 100.
MAX_LOGIT_SCALE = math.log(100.0)


class CachedTower(nn.Module):
    """Precomputed target vectors, looked up rather than recomputed. Holds no parameters."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.shape[-1] != self.dim:
            raise RuntimeError(
                f"cached captions are {vectors.shape[-1]}-dimensional, the "
                f"tower is {self.dim}. The cache was built for another tower.")
        return vectors.float()


class JointModel(nn.Module):
    """Image backbone and caption tower, compared in the tower's space."""

    def __init__(self, recipe: Recipe, *, device: str = "cuda") -> None:
        super().__init__()
        backbone, _, _, spec = load(recipe.backbone, device="cpu")
        # load() returns a model for measurement; here it is trained.
        backbone.requires_grad_(True)
        backbone.train()

        self.backbone = backbone
        self.spec = spec
        self.recipe = recipe
        self.reads_class_token = has_class_token(backbone)

        hidden = hidden_size(backbone)
        self.image_proj = nn.Linear(hidden, recipe.width, bias=False)
        nn.init.normal_(self.image_proj.weight, std=hidden ** -0.5)
        self.logit_scale_raw = nn.Parameter(torch.tensor(INITIAL_LOGIT_SCALE))

        self.tower = CachedTower(recipe.width)
        self.to(device)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Projected, normalised class token, or patch mean where there is no class token."""
        out = self.backbone(pixel_values=images).last_hidden_state
        summary = out[:, 0] if self.reads_class_token else out.mean(dim=1)
        return F.normalize(self.image_proj(summary.float()), dim=-1)

    def encode_text(self, captions) -> torch.Tensor:
        return F.normalize(self.tower(captions), dim=-1)

    def logit_scale(self) -> torch.Tensor:
        return self.logit_scale_raw.clamp(0.0, MAX_LOGIT_SCALE).exp()

    @torch.no_grad()
    def clamp_logit_scale(self) -> None:
        """Hold the raw temperature parameter inside its range after each step."""
        self.logit_scale_raw.clamp_(0.0, MAX_LOGIT_SCALE)

    def parameter_groups(self) -> list[dict]:
        """The backbone at one learning rate, the projection and temperature at another."""
        head = [p for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("backbone.")]
        body = [p for n, p in self.named_parameters()
                if p.requires_grad and n.startswith("backbone.")]
        return [{"name": "backbone", "params": body,
                 "lr": self.recipe.lr_backbone},
                {"name": "head", "params": head, "lr": self.recipe.lr_head}]

    def backbone_state(self) -> dict:
        """The payload :mod:`src.finetune.checkpoint` reads back."""
        state = {f"image_backbone.{k}": v.detach().cpu()
                 for k, v in self.backbone.state_dict().items()}
        state["image_proj.weight"] = self.image_proj.weight.detach().cpu()
        state["logit_scale"] = self.logit_scale_raw.detach().cpu()
        return state
