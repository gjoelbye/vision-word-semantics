"""The SPEECH-COCO training corpus: shards, split, text filter and caption cache.

The shards are separated into train and validation by photograph, every sample
whose photograph the text scan found writing in is excluded, and the caption
targets are read from a precomputed cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..config import CAPTIONS, CORPUS, DATA_ROOT

#: The SPEECH-COCO parquet export, which holds the photographs, captions and recordings.
SOURCE = DATA_ROOT / "spokencoco_hf" / "data"

#: The webdataset shards. The directory and shard names are part of
#: :func:`fingerprint`, which every checkpoint provenance records.
SHARDS = DATA_ROOT / "spokencoco-wds-288"

#: The records the shard and scan stages write under ``CORPUS``.
SPEECH_COCO_MANIFEST = CORPUS / "speech-coco_manifest.parquet"
SPEECH_COCO_PROVENANCE = CORPUS / "speech-coco_provenance.json"
SPEECH_COCO_SCAN = CORPUS / "speech-coco_text_scan.jsonl"
SPEECH_COCO_EXCLUDED = CORPUS / "speech-coco_excluded_keys.txt"

#: ImageNet channel statistics, which the processor of every finetuned backbone uses.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def described() -> str:
    """What a provenance record calls the corpus and its filter."""
    return "SPEECH-COCO, text-filtered"


def shards() -> list[str]:
    found = sorted(str(p) for p in SHARDS.glob("*.tar"))
    if not found:
        raise SystemExit(f"no .tar shards under {SHARDS}. Run "
                         f"scripts/finetune/speech_coco.py --shards.")
    return found


def split() -> tuple[list[str], list[str]]:
    """``(train, validation)`` shard paths, told apart by name."""
    found = shards()
    train = [s for s in found if "-train-" in Path(s).name]
    val = [s for s in found if "-val-" in Path(s).name]
    if not train or not val:
        raise SystemExit("the shards are not separated into train and val by "
                         "name; rewrite them with "
                         "scripts/finetune/speech_coco.py --shards")
    return train, val


def fingerprint() -> str:
    """sha256 of the shard file names, in order."""
    return hashlib.sha256(
        "\n".join(Path(s).name for s in shards()).encode()).hexdigest()


def corpus_size() -> int:
    """Samples in the corpus, as the shard stage recorded them."""
    record = SPEECH_COCO_PROVENANCE
    if not record.exists():
        raise SystemExit(f"no corpus provenance at {record}; run "
                         f"scripts/finetune/speech_coco.py --shards")
    return int(json.loads(record.read_text())["scan_rows"])


def excluded_keys(path: Path | None = None) -> set[str]:
    """Sample keys whose photograph the text scan found writing in."""
    path = path or SPEECH_COCO_EXCLUDED
    if not path.exists():
        raise SystemExit(f"no exclude list at {path}. Run "
                         f"scripts/finetune/speech_coco.py --scan.")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def kept_size() -> int:
    """Samples a run trains on after the text filter; an epoch is counted against it."""
    return corpus_size() - len(excluded_keys())


def transform(image_size: int, train: bool):
    """Image preprocessing: a random crop of 90 to 100 percent of the frame for
    training, a center crop for validation, both bicubic."""
    from torchvision import transforms as T

    resample = T.InterpolationMode.BICUBIC
    steps = ([T.RandomResizedCrop(image_size, scale=(0.9, 1.0),
                                  interpolation=resample)] if train
             else [T.Resize(image_size, interpolation=resample),
                   T.CenterCrop(image_size)])
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


class CaptionCache:
    """Float16 caption vectors on disk, addressed by sample key."""

    def __init__(self, tower: str, root: Path | None = None) -> None:
        base = (root or CAPTIONS) / "speech-coco" / tower
        meta_path = base / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"no caption cache for {tower} at {base}. Run "
                             f"scripts/finetune/caption_targets.py.")
        self.meta = json.loads(meta_path.read_text())
        self.dim = int(self.meta["dim"])
        self.vectors = np.load(base / "vectors.npy", mmap_mode="r")
        self.index = {k: i for i, k in enumerate(
            (base / "keys.txt").read_text().split("\n")) if k}

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def get(self, key: str) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(self.vectors[self.index[key]], dtype=np.float32))


def pipeline(recipe, *, train: bool, cache: CaptionCache, exclude: set[str],
             workers: int = 8, shuffle: int = 5000):
    """A loader of ``(images, caption vectors)`` micro-batches.

    Infinite, shuffled and resampled for training; finite and ordered for
    validation, so every validation scores the same samples.
    """
    import webdataset as wds

    train_shards, val_shards = split()
    chosen = train_shards if train else val_shards
    to_image = transform(recipe.image_size, train)

    def keep(sample: dict) -> bool:
        key = sample.get("__key__", "")
        if key in exclude or key not in cache:
            return False
        return bool(sample.get("txt", "").strip())

    def convert(sample: dict):
        return (to_image(sample["jpg"].convert("RGB")),
                cache.get(sample["__key__"]))

    def collate(samples: list):
        return (torch.stack([s[0] for s in samples]),
                torch.stack([s[1] for s in samples]))

    dataset = wds.WebDataset(
        chosen, shardshuffle=train, nodesplitter=wds.split_by_node,
        handler=wds.warn_and_continue, resampled=train, seed=recipe.seed,
        empty_check=False)
    if train:
        dataset = dataset.shuffle(shuffle)
    dataset = (dataset.decode("pil", handler=wds.warn_and_continue)
               .select(keep)
               .map(convert, handler=wds.warn_and_continue)
               .batched(recipe.micro_batch, collation_fn=collate, partial=False))

    return torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=workers if train else min(workers, 2),
        pin_memory=True, persistent_workers=bool(workers))
