"""Registry of every network the project reads, and how each is loaded.

The sweep holds fourteen pretrained image backbones: the DINOv3, DINOv2 and MAE
ladders, Web-DINO-1B, CLIP-B and SigLIP2-B, and an untrained ViT-B. Each of ten
of them is also finetuned twice on SPEECH-COCO, against a frozen text encoder
and against a frozen speech encoder. ``ENCODERS`` lists those two encoders and
the other text encoders the word set is read with.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import SEED


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    kind: str
    family: str
    params_m: int
    dtype: torch.dtype = torch.float32
    extra: dict = field(default_factory=dict)


_SPECS = [
    ModelSpec("dinov2-b", "facebook/dinov2-base", "dino", "DINOv2", 86),
    ModelSpec("dinov2-l", "facebook/dinov2-large", "dino", "DINOv2", 304),
    ModelSpec("dinov2-g", "facebook/dinov2-giant", "dino", "DINOv2", 1136),
    ModelSpec("dinov3-b", "facebook/dinov3-vitb16-pretrain-lvd1689m", "dino", "DINOv3", 86),
    ModelSpec("dinov3-l", "facebook/dinov3-vitl16-pretrain-lvd1689m", "dino", "DINOv3", 300),
    ModelSpec("dinov3-h", "facebook/dinov3-vith16plus-pretrain-lvd1689m", "dino", "DINOv3", 840),
    ModelSpec("dinov3-7b", "facebook/dinov3-vit7b16-pretrain-lvd1689m", "dino", "DINOv3",
              6716, dtype=torch.bfloat16),
    ModelSpec("mae-b", "facebook/vit-mae-base", "mae", "MAE", 86),
    ModelSpec("mae-l", "facebook/vit-mae-large", "mae", "MAE", 304),
    ModelSpec("mae-h", "facebook/vit-mae-huge", "mae", "MAE", 632),
    ModelSpec("clip-b", "openai/clip-vit-base-patch16", "clip", "CLIP", 86),
    ModelSpec("siglip2-b", "google/siglip2-base-patch16-224", "siglip", "SigLIP2", 86),
    ModelSpec("webssl-dino-1b", "facebook/webssl-dino1b-full2b-224", "dino", "Web-DINO", 1000),
    ModelSpec("vit-random-b", "google/vit-base-patch16-224", "random", "Untrained", 86),
]

#: The fourteen pretrained backbones, in reading order.
DEFAULT_SWEEP = ("dinov3-b", "dinov3-l", "dinov3-h", "dinov3-7b",
                 "webssl-dino-1b",
                 "dinov2-b", "dinov2-l", "dinov2-g",
                 "mae-b", "mae-l", "mae-h",
                 "clip-b", "siglip2-b", "vit-random-b")

#: Frozen encoders as ``key: (repository, output width, kind)``. The three
#: Qwen3 models read the word set as text; Qwen3-0.6B and w2v-BERT 2.0 are also
#: the targets the finetunes are trained against.
ENCODERS: dict[str, tuple[str, int, str]] = {
    "qwen3-0p6b-frozen": ("Qwen/Qwen3-Embedding-0.6B", 1024, "qwen3"),
    "qwen3-4b-frozen": ("Qwen/Qwen3-Embedding-4B", 2560, "qwen3"),
    "qwen3-8b-frozen": ("Qwen/Qwen3-Embedding-8B", 4096, "qwen3"),
    "w2v-bert-2.0-frozen": ("facebook/w2v-bert-2.0", 1024, "speech"),
}

#: The encoders a finetune can be trained against, and the suffix each adds to
#: the finetuned model's key.
_TOWER_SUFFIX = {"qwen3-0p6b-frozen": "", "w2v-bert-2.0-frozen": "-w2vbert"}


def finetune_key(backbone: str, tower: str) -> str:
    """Registry key, and checkpoint directory name, of a SPEECH-COCO finetune."""
    if tower not in _TOWER_SUFFIX:
        raise KeyError(f"no finetune is trained against {tower!r}; "
                       f"have {sorted(_TOWER_SUFFIX)}")
    return f"{backbone}-scocoft{_TOWER_SUFFIX[tower]}"


#: The SPEECH-COCO finetunes as ``(backbone, tower, step)``. The step is the
#: training step the checkpoint records, and loading refuses a checkpoint whose
#: record differs.
_FINETUNES = [
    ("dinov3-b", "qwen3-0p6b-frozen", 1199),
    ("dinov3-b", "w2v-bert-2.0-frozen", 1308),
    ("dinov3-l", "qwen3-0p6b-frozen", 763),
    ("dinov3-l", "w2v-bert-2.0-frozen", 1199),
    ("dinov3-h", "qwen3-0p6b-frozen", 436),
    ("dinov3-h", "w2v-bert-2.0-frozen", 981),
    ("webssl-dino-1b", "qwen3-0p6b-frozen", 218),
    ("webssl-dino-1b", "w2v-bert-2.0-frozen", 545),
    ("dinov2-b", "qwen3-0p6b-frozen", 1199),
    ("dinov2-b", "w2v-bert-2.0-frozen", 1090),
    ("dinov2-l", "qwen3-0p6b-frozen", 981),
    ("dinov2-l", "w2v-bert-2.0-frozen", 981),
    ("dinov2-g", "qwen3-0p6b-frozen", 218),
    ("dinov2-g", "w2v-bert-2.0-frozen", 872),
    ("mae-b", "qwen3-0p6b-frozen", 1308),
    ("mae-b", "w2v-bert-2.0-frozen", 1199),
    ("mae-l", "qwen3-0p6b-frozen", 872),
    ("mae-l", "w2v-bert-2.0-frozen", 981),
    ("mae-h", "qwen3-0p6b-frozen", 763),
    ("mae-h", "w2v-bert-2.0-frozen", 872),
]

_BASE = {s.key: s for s in _SPECS}
for _backbone, _tower, _step in _FINETUNES:
    _twin = _BASE[_backbone]
    _SPECS.append(ModelSpec(
        key=finetune_key(_backbone, _tower), repo=_twin.repo, kind=_twin.kind,
        family=_twin.family, params_m=_twin.params_m, dtype=_twin.dtype,
        extra={"base": _backbone, "tower": _tower, "train_step": _step}))

MODELS: dict[str, ModelSpec] = {s.key: s for s in _SPECS}

#: The twenty SPEECH-COCO finetunes, in the sweep order of their backbones,
#: the text run before the speech run.
FINETUNES = tuple(finetune_key(b, t) for b, t, _ in _FINETUNES)

#: Pretrained backbone of each finetune.
TWIN_OF: dict[str, str] = {k: MODELS[k].extra["base"] for k in FINETUNES}

PRETTY = {"dinov2-b": "DINOv2-B", "dinov2-l": "DINOv2-L", "dinov2-g": "DINOv2-G",
          "dinov3-b": "DINOv3-B", "dinov3-l": "DINOv3-L", "dinov3-h": "DINOv3-H",
          "dinov3-7b": "DINOv3-7B", "mae-b": "MAE-B", "mae-l": "MAE-L",
          "mae-h": "MAE-H", "clip-b": "CLIP-B", "siglip2-b": "SigLIP2-B",
          "webssl-dino-1b": "Web-DINO-1B", "vit-random-b": "Untrained ViT-B",
          "qwen3-0p6b-frozen": "Qwen3-0.6B", "qwen3-4b-frozen": "Qwen3-4B",
          "qwen3-8b-frozen": "Qwen3-8B", "w2v-bert-2.0-frozen": "w2v-BERT 2.0"}
PRETTY |= {k: f"{PRETTY[TWIN_OF[k]]} + SPEECH-COCO "
              f"{'speech' if ENCODERS[MODELS[k].extra['tower']][2] == 'speech' else 'text'}"
           for k in FINETUNES}

#: Family and parameter count of each registered key, for grouping and sorting.
FAMILY_OF: dict[str, str] = {k: s.family for k, s in MODELS.items()}
PARAMS_M: dict[str, int] = {k: s.params_m for k, s in MODELS.items()}


def layers_and_width(key: str) -> tuple[int, int]:
    """``(transformer blocks, hidden width)`` from the model's cached Hugging Face config.

    ``key`` is a registered model or an encoder. For a model whose config nests
    a vision config, the vision config is read.
    """
    from transformers import AutoConfig
    from transformers.utils import logging

    repo = MODELS[key].repo if key in MODELS else ENCODERS[key][0]
    verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    try:
        cfg = AutoConfig.from_pretrained(repo, local_files_only=True)
    finally:
        logging.set_verbosity(verbosity)
    cfg = getattr(cfg, "vision_config", cfg)
    return int(cfg.num_hidden_layers), int(cfg.hidden_size)


def select_layer_indices(n_layers: int, k: int = 10) -> list[int]:
    """``k`` evenly spaced hidden-state indices, 0 through ``n_layers``.

    Index 0 is the embedding output and index ``n_layers`` the final block, as
    in the tuple ``output_hidden_states=True`` returns. Depths differ across
    models, so comparisons use relative depth, ``index / n_layers``.
    """
    return sorted(set(np.linspace(0, n_layers, k).round().astype(int).tolist()))


def has_class_token(model) -> bool:
    """Whether the backbone prepends a class token to its patch sequence.

    Decided from the parameters: SigLIP2 has no ``cls_token`` or
    ``class_embedding`` and outputs patches only.
    """
    return any("cls_token" in n or "class_embedding" in n
               for n, _ in model.named_parameters())


def hidden_size(model) -> int:
    """Width of the backbone's hidden states."""
    for attr in ("hidden_size", "hidden_dim", "embed_dim"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int):
            return v
    raise RuntimeError(f"cannot determine hidden size of {type(model).__name__}")


def load(key: str, device: str = "cuda"):
    """Return ``(model, processor, n_layers, spec)``, in eval mode with gradients off.

    The untrained ViT-B is built from its config with a fixed seed, MAE runs
    with its mask ratio at zero, and a finetune is its backbone's architecture
    and processor with the checkpoint's weights loaded over it.
    """
    from transformers import (AutoConfig, AutoImageProcessor, AutoModel,
                              CLIPModel, CLIPProcessor)

    spec = MODELS[key]
    if spec.kind == "clip":
        model = CLIPModel.from_pretrained(spec.repo, dtype=spec.dtype).vision_model
        proc = CLIPProcessor.from_pretrained(spec.repo).image_processor
    elif spec.kind == "random":
        cfg = AutoConfig.from_pretrained(spec.repo)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = AutoModel.from_config(cfg)
        proc = AutoImageProcessor.from_pretrained(spec.repo)
    elif spec.kind == "mae":
        cfg = AutoConfig.from_pretrained(spec.repo)
        cfg.mask_ratio = 0.0
        model = AutoModel.from_pretrained(spec.repo, config=cfg, dtype=spec.dtype)
        proc = AutoImageProcessor.from_pretrained(spec.repo)
    else:
        model = AutoModel.from_pretrained(spec.repo, dtype=spec.dtype)
        proc = AutoImageProcessor.from_pretrained(spec.repo)

    if spec.kind == "siglip":
        model = model.vision_model

    if "base" in spec.extra:
        from .config import CHECKPOINTS
        from .finetune.checkpoint import load_backbone_state
        record = load_backbone_state(
            CHECKPOINTS / spec.key / "best.slim.pt", model,
            expect_repo=spec.repo, expect_step=spec.extra["train_step"])
        print(f"  {spec.key}: {record['n_tensors']} tensors from step "
              f"{record['train_step']}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(model.encoder.layer) if hasattr(model, "encoder") and \
        hasattr(model.encoder, "layer") else _count_layers(model)
    return model, proc, n_layers, spec


def _count_layers(model) -> int:
    for path in ("encoder.layers", "layer", "layers", "blocks"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return len(obj)
        except AttributeError:
            continue
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "depth"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    raise RuntimeError(f"cannot determine depth of {type(model).__name__}")


def free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
