"""Lexical readouts behind Tables 1, 4, 5, 6 and 7.

Each of the 14 models is read at the final layer for partner retrieval,
neighbor purity on the noun and brand lists, purity within spelling clusters,
the centering share, the comparisons within an item and nearest-centroid
accuracy on the held-out half. Across depth it is read for noun purity and
capital retrieval. The 4 encoders are read for partner retrieval, neighbor
purity, the comparisons within an item and nearest-centroid accuracy, from
their similarity matrices and word vectors. Writes
``results/{retrieval,purity,letter_cluster,centering,items,latent_categories,
depth,encoder_lexical}.parquet``.
"""
from __future__ import annotations

import argparse
import gc
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.analysis import lexical
from src.analysis.directions import (HEADLINE, DegenerateTau, estimator,
                                     load_cache, manifest, rownorm,
                                     shared_mean_share, word_order)
from src.analysis.lexical import K
from src.config import LEXICON, RESULTS, SIMILARITY, ensure_dirs
from src.features import speech, text
from src.models import DEFAULT_SWEEP
from src.stats import (N_PERM, binomial_se, holm, sign_test,
                       vocabulary_spread)
from src.stimuli.render import fingerprint

#: The encoders in table order: the three text encoders, then the speech encoder.
ENCODERS = text.TEXT_MODELS + speech.SPEECH_MODELS
#: The estimator each kind of encoder is indexed under in ``similarity/index.parquet``.
ESTIMATOR_OF = {"text": "carrier_mean_96", "speech": "rendition_mean_96"}
#: Table 1's plural-or-base column pools these relations over their summed pairs.
POOLED = ("negation_base", "singular_plural")
#: Seed of the length-stratified null of nearest-centroid accuracy.
PERMUTATION_SEED = 0


def lexicon() -> SimpleNamespace:
    """Word lists, pairs, items and half splits, indexed into the word order."""
    words = pd.read_parquet(LEXICON / "words.parquet")
    pairs = pd.read_parquet(LEXICON / "pairs.parquet")
    vocab = list(word_order())
    index = {w: i for i, w in enumerate(vocab)}
    grids = {}
    for name in ("category", "brand"):
        frame = words[words.role == name].reset_index(drop=True)
        grids[name] = dict(rows=np.array([index[w] for w in frame.word]),
                           labels=frame.category.to_numpy(),
                           clusters=frame.ortho.to_numpy().astype(int),
                           bands=lexical.length_bands(frame.n_chars))
    blocks = lexical.build_items(pairs, words)
    missing = {w for frame in blocks.values()
               for column in ("anchor", "kept", "other")
               for w in frame[column] if w not in index}
    if missing:
        raise SystemExit(f"{len(missing)} item words are not rendered: "
                         f"{sorted(missing)[:5]}")
    splits = {dataset: lexical.dataset_frame(words, dataset, role,
                                             lexical.SPLIT_SEED)
              for dataset, role in lexical.DATASETS.items()}
    return SimpleNamespace(
        words=words, pairs=pairs, vocab=vocab, index=index, grids=grids,
        blocks=blocks, splits=splits,
        relations={r: list(zip(g.a, g.b)) for r, g in pairs.groupby("relation")},
        kind_of=dict(zip(pairs.relation, pairs.kind)),
        capitals=list(zip(*pairs[pairs.relation == "country_capital"]
                          [["a", "b"]].values.T)))


def items_rows(key: str, cosine: np.ndarray, S: np.ndarray,
               lex: SimpleNamespace) -> list[dict]:
    """Wins and z-scored cosines per block, and the concrete-abstract contrast."""
    index = lex.index
    baseline, spread = vocabulary_spread(cosine)
    rows = []
    for name, frame in lex.blocks.items():
        anchor = np.array([index[w] for w in frame.anchor])
        kept = np.array([index[w] for w in frame.kept])
        other = np.array([index[w] for w in frame.other])
        difference = cosine[anchor, kept] - cosine[anchor, other]
        kept_z = (cosine[anchor, kept] - baseline) / spread
        other_z = (cosine[anchor, other] - baseline) / spread
        rows.append(dict(
            model=key, record_type="head_to_head", block=name,
            **sign_test(difference),
            kept_above_baseline=float(kept_z.mean()),
            kept_above_baseline_se=float(
                kept_z.std(ddof=1) / np.sqrt(len(kept_z))),
            other_above_baseline=float(other_z.mean()),
            other_above_baseline_se=float(
                other_z.std(ddof=1) / np.sqrt(len(other_z)))))

    # Synonyms within the ten nearest in either direction, by the anchor's class.
    block1 = lex.blocks["synonym_vs_twin"]
    records = block1.to_dict("records")
    found = np.zeros(len(records), bool)
    for left, right in (("anchor", "kept"), ("kept", "anchor")):
        found |= np.array([
            lexical.partner_rank(S, index[row[left]], index[row[right]]) < K
            for row in records])
    concrete = (block1["class"] == "concrete").to_numpy()
    table = [[int((found & concrete).sum()), int((~found & concrete).sum())],
             [int((found & ~concrete).sum()), int((~found & ~concrete).sum())]]
    rows.append(dict(
        model=key, record_type="class_contrast", block="synonym_vs_twin",
        concrete_found=table[0][0], abstract_found=table[1][0],
        n_concrete=int(concrete.sum()), n_abstract=int((~concrete).sum()),
        fisher_p=float(fisher_exact(table, alternative="greater")[1])))
    return rows


def latent_rows(key: str, T: np.ndarray, lex: SimpleNamespace) -> list[dict]:
    """Nearest-centroid accuracy on the held-out half of each list, with its null.

    The null shuffles the held-out labels within spelling cluster crossed with
    length band, with the predictions fixed.
    """
    rows = []
    for dataset, frame in lex.splits.items():
        label_order = np.array(sorted(frame.label.unique()))
        vocab_rows = np.array([lex.index[w] for w in frame.word])
        X = np.asarray(T[vocab_rows], dtype=np.float32)
        train = frame.split.eq("train").to_numpy()
        test = frame.split.eq("test").to_numpy()
        y_train = frame.loc[train, "label"].to_numpy()
        y_test = frame.loc[test, "label"].to_numpy()
        ortho_test = frame.loc[test, "ortho"].to_numpy()
        band_test = frame.loc[test, "length_band"].to_numpy()
        prediction = lexical.nearest_centroid_prediction(
            X[train], y_train, X[test], label_order)
        accuracy = float(np.mean(prediction == y_test))
        strata = np.unique(np.stack([ortho_test, band_test], axis=1), axis=0)
        rng = np.random.default_rng([PERMUTATION_SEED, 1])
        null = np.empty(N_PERM, dtype=float)
        for repeat in range(N_PERM):
            shuffled = y_test.copy()
            for ortho, band in strata:
                group = (ortho_test == ortho) & (band_test == band)
                shuffled[group] = rng.permutation(shuffled[group])
            null[repeat] = float(np.mean(prediction == shuffled))
        correct, held = int((prediction == y_test).sum()), int(test.sum())
        rows.append(dict(
            model=key, dataset=dataset, n_test=held, n_correct=correct,
            full_classification_accuracy=accuracy,
            full_classification_accuracy_se=binomial_se(correct, held),
            classification_chance=float(1 / len(label_order)),
            length_stratified_full_classification_accuracy_p=(
                (int((null >= accuracy).sum()) + 1) / (N_PERM + 1)),
            n_perm=N_PERM))
    return rows


def final_layer(key: str, lex: SimpleNamespace) -> dict[str, list[dict]]:
    """Every final-layer readout of one model, from one read of its cache."""
    _, layers, n_layers = load_cache(key)
    if int(layers[-1]) != n_layers:
        raise SystemExit(f"{key}: the last cached layer is {int(layers[-1])}, "
                         f"not the final layer {n_layers}")
    out = {name: [] for name in ("retrieval", "purity", "letter_cluster",
                                 "centering", "items", "latent_categories")}
    # The centering share is computed on the uncentered word directions, which
    # are then centered as ``estimator(HEADLINE)`` centers them.
    base = estimator("residual")(key, -1)
    out["centering"].append(dict(model=key,
                                 shared_mean_share=shared_mean_share(base)))
    base = base - base.mean(0, keepdims=True)
    T = rownorm(base)
    S = lexical.similarity(T)

    for relation, plist in lex.relations.items():
        for row in lexical.retrieval(S, lex.index, plist, relation):
            out["retrieval"].append(dict(model=key, kind=lex.kind_of[relation],
                                         **row))

    for axis, grid in lex.grids.items():
        Sg = lexical.similarity(T[grid["rows"]])
        labels = grid["labels"]
        perm = lexical.purity_permutation(Sg, labels, k=K,
                                          strata=grid["bands"])
        sub = lexical.purity_subsample(Sg, labels, k=K)
        out["purity"].append(dict(
            model=key, axis=axis, n=len(labels), purity=perm["purity"],
            se=sub["se"], p=perm["p"], chance=perm["chance"],
            n_perm=perm["n_perm"]))
        if axis == "category":
            out["letter_cluster"].append(dict(
                model=key, axis=axis,
                within_cluster=lexical.knn_purity(Sg, labels,
                                                  restrict=grid["clusters"]),
                within_cluster_chance=lexical.restricted_chance_purity(
                    labels, grid["clusters"])))

    out["items"] = items_rows(key, T @ T.T, S, lex)
    out["latent_categories"] = latent_rows(key, T, lex)
    return out


def depth_rows(key: str, lex: SimpleNamespace) -> list[dict]:
    """Noun purity and forward capital retrieval at every cached depth.

    At a depth where every direction is zero the readouts are missing.
    """
    grid = lex.grids["category"]
    _, layers, n_layers = load_cache(key)
    rows = []
    for i, layer in enumerate(layers):
        record = dict(model=key, layer=int(layer), layer_index=i,
                      n_layers=int(n_layers),
                      relative_depth=round(float(layer) / n_layers, 4))
        try:
            T = rownorm(estimator(HEADLINE)(key, i))
        except DegenerateTau:
            rows.append(record | dict(purity=np.nan, purity_se=np.nan,
                                      retrieval=np.nan, retrieval10=np.nan,
                                      retrieval10_se=np.nan))
            continue
        S = lexical.similarity(T[grid["rows"]])
        ranks = lexical.partner_ranks(lexical.similarity(T), lex.index,
                                      lex.capitals)
        pure = lexical.purity_subsample(S, grid["labels"])
        rows.append(record | dict(
            purity=pure["purity"], purity_se=pure["se"],
            retrieval=float((ranks == 0).mean()),
            retrieval10=float((ranks < K).mean()),
            retrieval10_se=binomial_se(int((ranks < K).sum()), len(ranks))))
    return rows


def encoder_rows(key: str, index_row: pd.Series,
                 lex: SimpleNamespace) -> list[dict]:
    """The final-layer readouts of one encoder, from its similarity matrix and word vectors."""
    vocab, index = lex.vocab, lex.index
    kind = str(index_row.kind)
    if index_row.estimator != ESTIMATOR_OF.get(kind):
        raise SystemExit(f"{key} is indexed as {kind} {index_row.estimator}")
    cosine = np.load(SIMILARITY / f"{key}.npy")
    if cosine.shape != (len(vocab), len(vocab)):
        raise SystemExit(f"the {key} matrix is {cosine.shape}, not "
                         f"{len(vocab)} square")
    common = dict(model=key, estimator=ESTIMATOR_OF[kind])
    records = []

    S = cosine.copy()
    np.fill_diagonal(S, -np.inf)
    forward = {}
    for relation, plist in lex.relations.items():
        for row in lexical.retrieval(S, index, plist, relation):
            if row["direction"] == "forward":
                forward[relation] = (row["hits10"], row["n"])
            records.append(common | dict(
                readout="rate10", group=relation, direction=row["direction"],
                estimate=row["hits10"] / row["n"],
                se=binomial_se(row["hits10"], row["n"]), n=row["n"],
                m=np.nan, hits10=row["hits10"], chance=row["chance10"],
                method="binomial"))
    k = sum(forward[r][0] for r in POOLED)
    n = sum(forward[r][1] for r in POOLED)
    records.append(common | dict(
        readout="rate10", group="+".join(POOLED), direction="forward",
        estimate=k / n, se=binomial_se(k, n), n=n, m=np.nan, hits10=k,
        chance=K / (len(vocab) - 1), method="binomial"))

    for axis, grid in lex.grids.items():
        rows, labels = grid["rows"], grid["labels"]
        Sg = cosine[np.ix_(rows, rows)].copy()
        np.fill_diagonal(Sg, -np.inf)
        out = lexical.purity_subsample(Sg, labels, k=K)
        records.append(common | dict(
            readout="purity", group=axis, direction="",
            estimate=out["purity"], se=out["se"], n=out["n"], m=out["m"],
            hits10=pd.NA, chance=lexical.chance_purity(labels),
            method="subsample"))

    baseline, spread = vocabulary_spread(cosine)
    for name, frame in lex.blocks.items():
        anchor = np.array([index[w] for w in frame.anchor])
        kept = np.array([index[w] for w in frame.kept])
        other = np.array([index[w] for w in frame.other])
        difference = cosine[anchor, kept] - cosine[anchor, other]
        records.append(common | dict(
            readout="wins", group=name, direction="",
            estimate=float(int((difference > 0).sum())), se=np.nan,
            n=int(len(frame)), m=np.nan, hits10=pd.NA,
            chance=len(frame) / 2, method="count"))
        if name != "synonym_vs_twin":
            continue
        for readout, column in (("kept_above_baseline", kept),
                                ("other_above_baseline", other)):
            z = (cosine[anchor, column] - baseline) / spread
            records.append(common | dict(
                readout=readout, group=name, direction="",
                estimate=float(z.mean()),
                se=float(z.std(ddof=1) / np.sqrt(len(z))),
                n=int(len(z)), m=np.nan, hits10=pd.NA, chance=0.0,
                method="items"))

    block1 = lex.blocks["synonym_vs_twin"]
    item_rows = block1.to_dict("records")
    found = np.zeros(len(item_rows), bool)
    for left, right in (("anchor", "kept"), ("kept", "anchor")):
        found |= np.array([
            lexical.partner_rank(S, index[row[left]], index[row[right]]) < K
            for row in item_rows])
    for cls in ("concrete", "abstract"):
        take = (block1["class"] == cls).to_numpy()
        records.append(common | dict(
            readout="class_contrast", group=cls, direction="either",
            estimate=float(found[take].sum()), se=np.nan,
            n=int(take.sum()), m=np.nan, hits10=pd.NA, chance=np.nan,
            method="count"))

    # Nearest-centroid accuracy needs the centered, unit-length vectors the
    # matrix was built from.
    loader = speech if kind == "speech" else text
    vectors = loader.load(key, vocab)
    unit = rownorm(np.asarray(vectors - vectors.mean(0, keepdims=True),
                              np.float32))
    for dataset, frame in lex.splits.items():
        label_order = np.array(sorted(frame.label.unique()))
        X = unit[np.array([index[w] for w in frame.word])]
        accuracy, correct, held = lexical.nearest_centroid_accuracy(
            X, frame, label_order)
        records.append(common | dict(
            readout="classification", group=dataset, direction="",
            estimate=accuracy, se=binomial_se(correct, held), n=held,
            m=np.nan, hits10=pd.NA, chance=1 / len(label_order),
            method="binomial"))
    return records


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    lex = lexicon()
    sha = fingerprint(manifest())
    print(f"{len(lex.vocab):,} words; {len(DEFAULT_SWEEP)} models, "
          f"{len(ENCODERS)} encoders")

    tables = {name: [] for name in ("retrieval", "purity", "letter_cluster",
                                    "centering", "items", "latent_categories",
                                    "depth")}
    for key in DEFAULT_SWEEP:
        for name, rows in final_layer(key, lex).items():
            tables[name] += rows
        load_cache.cache_clear()
        gc.collect()
        tables["depth"] += depth_rows(key, lex)
        load_cache.cache_clear()
        gc.collect()
        print(f"  {key}")

    items = pd.DataFrame(tables["items"])
    # Holm over the 14 models, separately for each block.
    head = items.record_type.eq("head_to_head")
    for _, group in items[head].groupby("block"):
        items.loc[group.index, "sign_p_holm"] = holm(
            group.sign_p.fillna(1.0).to_numpy(float))
    tables["items"] = items

    word_sha = text.fingerprint(lex.vocab)
    index = pd.read_parquet(SIMILARITY / "index.parquet").set_index("model")
    encoders = []
    for key in ENCODERS:
        if key not in index.index or index.loc[key, "word_sha"] != word_sha:
            raise SystemExit(f"no similarity matrix of {key} for the current "
                             f"word set; run scripts/features/similarity.py")
        encoders += encoder_rows(key, index.loc[key], lex)
        print(f"  {key}")
    encoders = pd.DataFrame(encoders)
    encoders["hits10"] = encoders.hits10.astype("Int64")
    encoders["word_sha"] = word_sha
    tables["encoder_lexical"] = encoders

    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        frame["manifest_sha"] = sha
        frame.to_parquet(RESULTS / f"{name}.parquet", index=False)
        print(f"wrote results/{name}.parquet ({len(frame)} rows)")


if __name__ == "__main__":
    main()
