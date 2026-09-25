"""Readouts between class names and class photographs behind Tables 2, 8, 9 and 10.

For the 14 models it computes, at the final layer, the matched cosine
similarity between each class name's word direction and its class direction
with a permutation null, retrieval in both directions on all photographs and on
those with no detected text, the change when a name's own background is left
out, and per-class diagnostics; across depth, the matched cosine similarity and
word-to-class top-10 accuracy. Writes the five results/concept_*.parquet
tables: alignment, retrieval, checks, classes and depth.
"""
from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from src import concepts, nu
from src.config import BACKGROUNDS, LEXICON, RESULTS, ensure_dirs
from src.models import DEFAULT_SWEEP
from src.neighbors import K
from src.render import fingerprint
from src.slots import slot_table
from src.tau import (CONCEPT_IMAGES, CONCEPT_WORDS, HEADLINE, DegenerateTau,
                     estimator, load_cache, manifest, rownorm, word_order)


def referent_slots(words: pd.DataFrame) -> dict[str, list[int]]:
    """The slots whose background photograph shows the class a word names."""
    backgrounds = pd.read_parquet(BACKGROUNDS / "backgrounds.parquet")
    bg_wnid = {int(r.background_index): concepts.wnids()[int(r.imagenet_label)]
               for r in backgrounds.itertuples()}
    slot_of = {int(r.slot): bg_wnid.get(int(r.background_index))
               for r in slot_table().itertuples()}
    own_slot = {}
    for r in words.itertuples():
        hit = [s for s, w in slot_of.items() if w == r.wnid]
        if hit:
            own_slot[r.word] = hit
    return own_slot


def referent_check(key: str, T_full: np.ndarray, words: pd.DataFrame,
                   index: dict[str, int], own_slot: dict[str, list[int]]) -> dict:
    """Mean matched cosine of the affected words with every slot and without their own."""
    N = rownorm(nu.class_mean(key, -1, "all"))
    rows, target = nu.word_class_rows(words.word, words.wnid, "all")
    words_kept = list(words.word.iloc[rows])
    r_all = np.array([index[w] for w in words_kept])
    by_slot = {}
    for i, w in enumerate(words_kept):
        for sl in own_slot.get(w, []):
            by_slot.setdefault(int(sl), []).append(i)
    every = [int(x) for x in slot_table().slot]
    with_all, without, done = [], [], []
    for sl, members in sorted(by_slot.items()):
        other = tuple(x for x in every if x != sl)
        T_less = rownorm(estimator(HEADLINE)(key, -1, other, ds=CONCEPT_WORDS))
        for i in members:
            with_all.append(float((T_full[r_all[i]] * N[target[i]]).sum()))
            without.append(float((T_less[r_all[i]] * N[target[i]]).sum()))
            done.append(words_kept[i])
    return dict(model=key, n_words=len(done),
                max_backgrounds_per_word=max(len(own_slot[w]) for w in done),
                with_all_slots=float(np.mean(with_all)),
                without_own_slot=float(np.mean(without)))


def class_rows(key: str, words: pd.DataFrame) -> pd.DataFrame:
    """Coherence of each class's photographs and its nearest other class."""
    N = rownorm(nu.class_mean(key, -1, "all"))
    classes = list(nu.retained("all"))
    who, close = nu.nearest_class(N)
    named = dict(zip(words.wnid, words.word))
    return pd.DataFrame(dict(
        model=key, wnid=classes,
        word=[named.get(c, "") for c in classes],
        coherence=nu.class_coherence(key, -1, "all"),
        nearest_wnid=[classes[i] for i in who],
        nearest_word=[named.get(classes[i], "") for i in who],
        nearest_cosine=close))


def depth_rows(key: str, words: pd.DataFrame,
               index: dict[str, int]) -> list[dict]:
    """Matched cosine and word-to-class top-10 accuracy at cached depths 1 to 9."""
    rows, target = nu.word_class_rows(words.word, words.wnid, "all")
    rows = np.array([index[w] for w in words.word.iloc[rows]])
    _, layers, n_layers = load_cache(key, ds=CONCEPT_WORDS)
    out = []
    for i, layer in enumerate(layers):
        if i == 0:
            continue
        record = dict(model=key, layer=int(layer), layer_index=i,
                      n_layers=int(n_layers),
                      relative_depth=round(float(layer) / n_layers, 4))
        try:
            T = rownorm(estimator(HEADLINE)(key, i, ds=CONCEPT_WORDS))
            N = rownorm(nu.class_mean(key, i, "all"))
        except DegenerateTau:
            out.append(record | dict(alignment=np.nan,
                                     word_to_class_rate10=np.nan,
                                     word_to_class_chance10=np.nan))
            continue
        rank = nu.cross_rank(T[rows], N, target)
        out.append(record | dict(
            alignment=float((T[rows] * N[target]).sum(1).mean()),
            word_to_class_rate10=int((rank < K).sum()) / len(rank),
            word_to_class_chance10=K / len(N)))
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    words = pd.read_parquet(LEXICON / "concepts.parquet")
    vocab = list(word_order(CONCEPT_WORDS))
    index = {w: i for i, w in enumerate(vocab)}
    words = words[words.word.isin(index)].reset_index(drop=True)
    own_slot = referent_slots(words)
    print(f"{len(vocab)} class names; {len(own_slot)} on their own background")

    tables = {name: [] for name in ("concept_alignment", "concept_retrieval",
                                    "concept_checks", "concept_classes",
                                    "concept_depth")}
    for key in DEFAULT_SWEEP:
        T_full = rownorm(estimator(HEADLINE)(key, -1, ds=CONCEPT_WORDS))
        for subset in nu.SUBSETS:
            N = rownorm(nu.class_mean(key, -1, subset))
            rows, target = nu.word_class_rows(words.word, words.wnid, subset)
            rows = np.array([index[w] for w in words.word.iloc[rows]])
            tables["concept_alignment"].append(dict(
                model=key, subset=subset, n_classes=len(N),
                **nu.concept_alignment(T_full, N, rows, target)))
            for row in nu.concept_retrieval(T_full, N, rows, target):
                tables["concept_retrieval"].append(
                    dict(model=key, subset=subset, **row))
            del N
        tables["concept_checks"].append(
            referent_check(key, T_full, words, index, own_slot))
        tables["concept_classes"].append(class_rows(key, words))
        del T_full
        tables["concept_depth"] += depth_rows(key, words, index)
        load_cache.cache_clear()
        gc.collect()
        print(f"  {key}")

    word_sha = fingerprint(manifest(CONCEPT_WORDS))
    image_sha = fingerprint(manifest(CONCEPT_IMAGES))
    for name, rows in tables.items():
        frame = (pd.concat(rows, ignore_index=True) if name == "concept_classes"
                 else pd.DataFrame(rows))
        frame["word_manifest_sha"] = word_sha
        frame["image_manifest_sha"] = image_sha
        frame.to_parquet(RESULTS / f"{name}.parquet", index=False)
        print(f"wrote results/{name}.parquet ({len(frame)} rows)")


if __name__ == "__main__":
    main()
