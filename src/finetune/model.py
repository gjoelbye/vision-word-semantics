"""The image backbone joined to a cached caption tower, and the training objective.

The backbone is built by :func:`src.models.load`, the loader the embeddings
use, so the finetune differs from its twin in weights alone. Two
parameters join the two sides: a projection into the tower's space and a
learned temperature. Both are dropped when the checkpoint is read back.

The objective is symmetric InfoNCE over a large batch, assembled by gradient
caching. Each micro-batch is encoded once without gradients, the loss is taken
over the whole batch of vectors, and the gradient at each image vector is then
replayed through the backbone by re-encoding its micro-batch. The result is the
full-batch gradient at the memory cost of one micro-batch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import has_class_token, hidden_size, load

if TYPE_CHECKING:
    from .training import Recipe

#: Starting temperature as a log scale, CLIP's 1/0.07; the value is learned.
INITIAL_LOGIT_SCALE = math.log(1 / 0.07)

#: Ceiling on the learned log scale, so the scale is at most 100.
MAX_LOGIT_SCALE = math.log(100.0)


class CachedTower(nn.Module):
    """Precomputed caption vectors, returned as float32 after a width check. Holds no parameters."""

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
        # load() returns the model frozen and in eval mode.
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
        """Projected, normalized class token, or patch mean where there is no class token."""
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


def infonce(image: torch.Tensor, text: torch.Tensor,
            logit_scale: torch.Tensor) -> torch.Tensor:
    """Mean of the image-to-caption and caption-to-image cross-entropies.

    ``image`` and ``text`` are L2-normalized ``(batch, dim)``; row ``i`` of each
    is a true pair.
    """
    logits = logit_scale * image @ text.t()
    target = torch.arange(image.shape[0], device=image.device)
    return 0.5 * (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.t(), target))


def in_batch_accuracy(image: torch.Tensor, text: torch.Tensor) -> tuple[float, float]:
    """Top-1 retrieval within the batch, image to caption and caption to image."""
    sim = image @ text.t()
    target = torch.arange(sim.shape[0], device=sim.device)
    return (float((sim.argmax(dim=1) == target).float().mean()),
            float((sim.argmax(dim=0) == target).float().mean()))


@dataclass(frozen=True)
class StepResult:
    loss: float
    image_accuracy: float
    text_accuracy: float
    batch: int


def cached_step(model, batches, *, autocast_dtype: torch.dtype | None = None
                ) -> StepResult:
    """One optimizer step's gradients over ``batches`` micro-batches.

    The caption vectors come from a frozen cache and carry no gradient, so only
    the image side is re-encoded. The temperature receives its gradient from
    the full-batch loss alone.
    """
    if not batches:
        raise ValueError("cached_step needs at least one micro-batch")

    cached_image, cached_text = [], []
    for images, captions in batches:
        with torch.no_grad(), _autocast(autocast_dtype):
            image = model.encode_image(images)
            text = model.encode_text(captions)
        cached_image.append(image.detach().float().requires_grad_(True))
        cached_text.append(text.detach().float())

    image = torch.cat(cached_image)
    text = torch.cat(cached_text)
    loss = infonce(image, text, model.logit_scale())
    loss.backward()
    accuracy = in_batch_accuracy(image.detach(), text.detach())

    for i, (images, _) in enumerate(batches):
        with _autocast(autocast_dtype):
            surrogate = (model.encode_image(images).float()
                         * cached_image[i].grad).sum()
        surrogate.backward()

    return StepResult(float(loss.detach()), accuracy[0], accuracy[1],
                      image.shape[0])


def _autocast(dtype: torch.dtype | None):
    """Autocast on CUDA with the weight cache off, or nothing on the CPU."""
    from contextlib import nullcontext

    if dtype is None or not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False)
