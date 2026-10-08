"""Symmetric InfoNCE over a large batch, assembled by gradient caching.

Each micro-batch is encoded once without gradients, the loss is taken over the
whole batch of vectors, and the gradient at each image vector is then replayed
through the backbone by re-encoding its micro-batch. The result is the
full-batch gradient at the memory cost of one micro-batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
