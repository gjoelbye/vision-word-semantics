"""The settings of one finetuning run, and the schedule derived from the corpus.

A recipe holds everything that distinguishes one run from another and nothing
about the machine. It is written into the checkpoint and its provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import SEED
from ..models import ENCODERS, finetune_key

#: Images per forward pass for each backbone, the largest that fit a 32 GB card.
#: The gradient does not depend on it.
MICRO_BATCH: dict[str, int] = {
    "dinov3-b": 96, "dinov2-b": 96, "mae-b": 96,
    "dinov3-l": 48, "dinov2-l": 48, "mae-l": 48,
    "mae-h": 24, "dinov3-h": 16, "webssl-dino-1b": 12, "dinov2-g": 8,
}

#: Ceiling on optimizer steps, and on passes over the filtered corpus.
STEP_CEILING = 10_000
MAX_EPOCHS = 3.0

#: Ceiling on the warmup length and on the validation and checkpoint intervals.
CADENCE_CEILING = 500

#: Validations without improvement before a run stops early.
PATIENCE = 3

#: Images per optimizer step.
BATCH = 1056


def schedule(kept: int, batch: int = BATCH) -> dict[str, int]:
    """Steps, warmup, validation interval and checkpoint interval for a corpus of ``kept`` samples.

    The run is ``MAX_EPOCHS`` passes, capped at ``STEP_CEILING`` steps; warmup
    is a tenth of it, validation every twelfth and a checkpoint every sixth,
    each at most ``CADENCE_CEILING`` and at least 10 steps.
    """
    steps = min(STEP_CEILING, max(1, int(MAX_EPOCHS * kept / batch)))
    return {"steps": steps,
            "warmup": min(CADENCE_CEILING, max(10, steps // 10)),
            "validate_every": min(CADENCE_CEILING, max(10, steps // 12)),
            "checkpoint_every": min(CADENCE_CEILING, max(10, steps // 6))}


@dataclass(frozen=True)
class Recipe:
    backbone: str
    tower: str
    steps: int
    warmup: int
    micro_batch: int
    batch: int = BATCH
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    image_size: int = 224
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.batch % self.micro_batch:
            raise ValueError(
                f"batch {self.batch} is not a whole number of micro-batches of "
                f"{self.micro_batch}")
        if self.warmup >= self.steps:
            raise ValueError("warmup must be shorter than the run")
        finetune_key(self.backbone, self.tower)

    @property
    def accumulation(self) -> int:
        """Micro-batches per optimizer step."""
        return self.batch // self.micro_batch

    @property
    def width(self) -> int:
        """Dimension the image is projected into, the tower's output width."""
        return ENCODERS[self.tower][1]

    @property
    def name(self) -> str:
        """Registry key of the finetuned model."""
        return finetune_key(self.backbone, self.tower)

    def to_dict(self) -> dict:
        return asdict(self) | {"accumulation": self.accumulation,
                               "width": self.width, "name": self.name}
