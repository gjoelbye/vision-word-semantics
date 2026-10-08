"""Clue gains, bootstrap intervals and sign-flip tests for the two-clue items.

Reads the two-clue manifest of ``scripts/stimuli/two_clue.py`` and
the final-layer class-token caches of the 14 image models written by
``scripts/features/embed.py --manifest two_clue``, and scores the 12
confirmatory and the 6 pilot clue grids as separate analysis sets, with Holm
correction across the 13 pretrained models within each set. Writes
``results/two_clue_summary.parquet``.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.analysis.two_clue import (
    READOUTS,
    condition_vectors,
    target_metrics,
    validate_dense_blocks,
)
from src.config import RESULTS
from src.models import DEFAULT_SWEEP
from src.stats import bootstrap_mean_ci, exact_sign_flip_p, holm
from src.stimuli.two_clue_render import (
    EMBEDDINGS_ROOT,
    MANIFEST_META_PATH,
    MANIFEST_PATH,
    load_curated_items,
    manifest_fingerprint,
)

BOOTSTRAP_SEED = 20260828
N_BOOTSTRAP = 20_000
OUT = RESULTS / "two_clue_summary.parquet"

#: The analysis sets, each scored on its own clue grids.
ANALYSIS_SETS = ("confirmatory", "pilot")

#: Grid-level metrics carried into the summary.
METRICS = (
    "joint_margin",
    "interaction_margin",
    "feature_axis_gain_over_base",
    "base_axis_gain_over_feature",
)

#: Metrics reported with a percentile bootstrap interval.
INTERVAL_METRICS = (
    "interaction_margin",
    "feature_axis_gain_over_base",
    "base_axis_gain_over_feature",
)

#: The model excluded from the Holm family.
UNTRAINED = "vit-random-b"


def validate_manifest(manifest: pd.DataFrame, manifest_sha: str) -> pd.DataFrame:
    """One row per item, after checking the manifest against its metadata."""
    required = {
        "equation_id", "split", "family_id", "feature", "base", "target",
        "slot", "replicate", "condition", "image_path",
    }
    if missing := required - set(manifest):
        raise RuntimeError(f"two-clue manifest lacks {sorted(missing)}")
    validate_dense_blocks(manifest, n_equations=144, n_slots=96)
    meta = json.loads(MANIFEST_META_PATH.read_text())
    if (meta.get("manifest_sha") != manifest_sha
            or meta.get("render_complete") is not True
            or meta.get("rows") != len(manifest)):
        raise RuntimeError("two-clue manifest metadata is stale or incomplete")
    return (manifest.groupby("equation_id", sort=False).first().reset_index()[
        ["equation_id", "family_id", "split", "feature", "base", "target"]])


def load_cache(key: str, manifest_sha: str, n_rows: int) -> np.ndarray:
    """Final-layer class tokens of one model, refused unless complete and current."""
    path = EMBEDDINGS_ROOT / f"{key}.cls.npy"
    meta_path = EMBEDDINGS_ROOT / f"{key}.meta.json"
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing two-clue cache for {key}")
    meta = json.loads(meta_path.read_text())
    checks = {
        "sha": meta.get("sha") == manifest_sha,
        "done": meta.get("done") is True,
        "cursor": meta.get("cursor") == n_rows,
        "layer_mode": meta.get("layer_mode") == "final",
        "final_layer": meta.get("layers") == [meta.get("n_layers")],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"{key} cache failed: {', '.join(failed)}")
    cache = np.load(path, mmap_mode="r")
    if cache.shape != (n_rows, 1, meta.get("dim")) or cache.dtype != np.float16:
        raise RuntimeError(f"unexpected {key} array {cache.shape} {cache.dtype}")
    return cache


def model_vectors(cache: np.ndarray, manifest: pd.DataFrame,
                  items: pd.DataFrame) -> dict[str, np.ndarray]:
    """The difference vectors of every item, each averaged over the 96 slots."""
    full = {name: [] for name in (*READOUTS, "target")}
    for equation_index, equation_id in enumerate(items.equation_id):
        lo, hi = equation_index * 96 * 6, (equation_index + 1) * 96 * 6
        rows = manifest.iloc[lo:hi]
        if rows.equation_id.nunique() != 1 or rows.equation_id.iloc[0] != equation_id:
            raise RuntimeError("equation rows are not contiguous in the manifest")
        block = np.asarray(cache[lo:hi, 0, :], dtype=np.float32).reshape(
            96, 6, -1)
        for name, value in condition_vectors(block).items():
            full[name].append(value)
    return {name: np.stack(values) for name, values in full.items()}


def family_table(items: pd.DataFrame) -> pd.DataFrame:
    """Grid means of the metrics, one row per model and clue grid."""
    keys = ["model", "family_id"]
    grouped = items.groupby(keys, sort=False)
    if not grouped.split.nunique().eq(1).all():
        raise RuntimeError("a family spans both splits")
    values = grouped[list(METRICS)].mean().reset_index()
    split = grouped[["split"]].first().reset_index()
    return values.merge(split, on=keys, validate="one_to_one")


def summarize(families: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Means, intervals and p-values over the clue grids of one analysis set.

    The bootstrap draws are taken once and shared by every model and metric.
    """
    rows = []
    n_families = families.family_id.nunique()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n_families, size=(N_BOOTSTRAP, n_families))
    for model in models:
        frame = families[families.model.eq(model)].sort_values(
            "family_id", kind="mergesort")
        if len(frame) != n_families:
            raise RuntimeError(f"{model} has {len(frame)} of {n_families} families")
        row = {
            "model": model,
            "n_families": n_families,
            "n_equations": n_families * 8,
        }
        for metric in METRICS:
            row[metric] = float(frame[metric].to_numpy(float).mean())
        for metric in INTERVAL_METRICS:
            row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = (
                bootstrap_mean_ci(frame[metric].to_numpy(float), draws))
        for metric in ("joint_margin", "feature_axis_gain_over_base",
                       "base_axis_gain_over_feature"):
            row[f"{metric}_p_greater"] = exact_sign_flip_p(
                frame[metric].to_numpy(float))
        # The interaction margin has no predicted sign, so its test is two-sided.
        row["interaction_margin_p_two_sided"] = exact_sign_flip_p(
            frame["interaction_margin"].to_numpy(float), alternative="two-sided")
        row["structured_iut_effect"] = min(
            row["feature_axis_gain_over_base"],
            row["base_axis_gain_over_feature"])
        row["structured_iut_p_greater"] = max(
            row["feature_axis_gain_over_base_p_greater"],
            row["base_axis_gain_over_feature_p_greater"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    pretrained = ~summary.model.eq(UNTRAINED)
    for source, destination in (
        ("joint_margin_p_greater", "joint_margin_p_holm"),
        ("structured_iut_p_greater", "structured_iut_p_holm"),
        ("interaction_margin_p_two_sided", "interaction_margin_p_holm"),
    ):
        summary[destination] = np.nan
        summary.loc[pretrained, destination] = holm(
            summary.loc[pretrained, source].to_numpy(float))
    return summary


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    models = list(DEFAULT_SWEEP)

    curated, _, dataset_sha = load_curated_items()
    manifest = pd.read_parquet(MANIFEST_PATH).reset_index(drop=True)
    manifest_sha = manifest_fingerprint(manifest)
    items = validate_manifest(manifest, manifest_sha)
    expected = curated[["family_id", "feature", "base", "target"]]
    got = items[["family_id", "feature", "base", "target"]]
    if set(map(tuple, expected.to_numpy())) != set(map(tuple, got.to_numpy())):
        raise RuntimeError("manifest equations differ from the curated tables")

    scored = []
    for model in models:
        cache = load_cache(model, manifest_sha, len(manifest))
        frame = target_metrics(items, model_vectors(cache, manifest, items))
        frame.insert(0, "model", model)
        scored.append(frame)
        print(f"  scored {model}", flush=True)
    family = family_table(pd.concat(scored, ignore_index=True))

    summaries = []
    for name in ANALYSIS_SETS:
        subset = family[family.split.eq(name)]
        if subset.empty:
            raise RuntimeError(f"analysis set {name!r} is empty")
        part = summarize(subset, models)
        part.insert(1, "analysis_set", name)
        summaries.append(part)
    summary = pd.concat(summaries, ignore_index=True).assign(
        n_bootstrap=N_BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED,
        dataset_sha=dataset_sha, manifest_sha=manifest_sha)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT, index=False)
    print(summary[summary.analysis_set.eq("confirmatory")][[
        "model", "feature_axis_gain_over_base", "base_axis_gain_over_feature",
        "structured_iut_p_holm", "joint_margin_p_holm", "interaction_margin",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"wrote results/{OUT.name}")


if __name__ == "__main__":
    main()
