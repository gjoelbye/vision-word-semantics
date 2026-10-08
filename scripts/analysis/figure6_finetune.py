"""Measure noun purity, partner retrieval and split-half cosine before and after finetuning.

The models are the ten finetuned backbones, their twenty SPEECH-COCO finetunes
and CLIP-B. Word directions come from 4 images per word, the four backgrounds
of one typography setting drawn at random five times, and from all 96. For each
model and image count, ``results/finetune_effect.parquet`` holds the mean
(``value``) and standard deviation (``spread``) over the draws of three
readouts: noun purity among the 10 nearest nouns (``purity``, ``common_noun``);
top-10 accuracy for country-capital and for synonym pairs, each pair queried in
both directions (``retrieval``, with the relation as task); and the split-half
cosine (``reliability``, ``tau``, k 0). ``results/finetune_draws.parquet``
lists the typography setting of every draw.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

from src.analysis import directions, lexical
from src.config import LEXICON, RESULTS, ensure_dirs
from src.models import FINETUNES, TWIN_OF
from src.stimuli.render import fingerprint
from src.stimuli.typography import FACTORS, REPLICATES, slot_table, slots_for

#: The ten finetuned backbones, their finetunes, and CLIP-B for reference.
MODELS = tuple(dict.fromkeys(TWIN_OF[k] for k in FINETUNES)) + FINETUNES + ("clip-b",)

#: Word images per word: one typography setting, and all 96.
IMAGE_COUNTS = (4, 96)

#: Random draws of the typography settings; the full set needs one.
DRAWS = 5
SEED = 0

#: Neighbors counted for retrieval and purity.
K = 10

#: The retrieval relations.
RELATIONS = ("country_capital", "synonym")


def retrieval_tasks() -> dict[str, list[tuple[str, str]]]:
    """Directed pairs per relation, each pair once in each direction."""
    pairs = pd.read_parquet(LEXICON / "pairs.parquet")

    def directed(selector):
        d = pairs[selector]
        return list(zip(d.a, d.b)) + list(zip(d.b, d.a))

    return {name: directed(pairs.relation == name) for name in RELATIONS}


def measure(models) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The readouts of every model at every image count, and the draws they used."""
    words = pd.read_parquet(LEXICON / "words.parquet")
    index = {w: i for i, w in enumerate(directions.word_order())}
    nouns = words[words.role == "category"].reset_index(drop=True)
    noun_rows = np.array([index[w] for w in nouns.word])
    labels = nouns.category.to_numpy()
    tasks = retrieval_tasks()
    appearances = sorted(slot_table().appearance.unique())
    rng = random.Random(SEED)
    rows, drawn = [], []
    for key in models:
        for images in IMAGE_COUNTS:
            n_app = images // REPLICATES
            draws = 1 if n_app == len(appearances) else DRAWS
            got = {}
            for draw in range(draws):
                rng.seed(SEED + draw)
                chosen = sorted(rng.sample(appearances, n_app))
                if key == models[0]:
                    drawn += [dict(images=images, draw=draw, appearance=int(a))
                              for a in chosen]
                T = directions.rownorm(
                    directions.estimator(directions.HEADLINE)(
                        key, -1, slots_for(chosen)))
                similarity = lexical.similarity(T)
                for name, pairs in tasks.items():
                    ranks = lexical.partner_ranks(similarity, index, pairs)
                    got.setdefault(("retrieval", name, K), []).append(
                        float((ranks < K).mean()))
                got.setdefault(("purity", "common_noun", K), []).append(
                    lexical.knn_purity(lexical.similarity(T[noun_rows]),
                                       labels, K))
                got.setdefault(("reliability", "tau", 0), []).append(
                    directions.split_half_cosine(key, chosen))
            for (readout, task, k), values in got.items():
                rows.append(dict(model=key, readout=readout, task=task, k=int(k),
                                 images=int(images), draws=int(draws),
                                 value=float(np.mean(values)),
                                 spread=float(np.std(values))))
            print(f"  {key:26s} {images:>3} images", flush=True)
        directions.load_cache.cache_clear()

    sha = fingerprint(directions.manifest())
    effect = pd.DataFrame(rows).assign(
        manifest_sha=sha, representation="cls",
        estimator=directions.HEADLINE, layer_index=-1)
    setting = (slot_table().drop_duplicates("appearance")
               .set_index("appearance")[["font", "placement", "size", "color", "box"]])
    assert set(setting.columns) == set(FACTORS)
    draws = pd.DataFrame(drawn).join(setting, on="appearance").assign(
        manifest_sha=sha)
    return effect, draws


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    effect, draws = measure(MODELS)
    effect.to_parquet(RESULTS / "finetune_effect.parquet", index=False)
    draws.to_parquet(RESULTS / "finetune_draws.parquet", index=False)
    print(f"wrote results/finetune_effect.parquet ({len(effect)} rows) and "
          f"results/finetune_draws.parquet ({len(draws)} rows)")


if __name__ == "__main__":
    main()
