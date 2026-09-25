"""Layout, manifest and rendering of the two-clue items.

Each item is rendered in six conditions on the 96 slots of the main word set,
under ``composition/``. The position of the broad clue and of the target cross
2 by 2 over the four background replicates.
"""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .composition_items import (dataset_fingerprint, read_table,
                                require_valid, validate_datasets)
from .config import (BACKGROUNDS, COMPOSITION, WORD_LISTS, local_path,
                     manifest_path)
from .slots import slot_table
from .typography import (Box, box_at_centre, crop_safe_window,
                         legible_centre, render_overlay, render_scramble,
                         text_extent)


RENDERS_ROOT = COMPOSITION / "renders"
EMBEDDINGS_ROOT = COMPOSITION / "embeddings"
LAYOUT_PATH = COMPOSITION / "layout.parquet"
LAYOUT_META_PATH = COMPOSITION / "layout.meta.json"
MANIFEST_PATH = COMPOSITION / "manifest.parquet"
MANIFEST_META_PATH = COMPOSITION / "manifest.meta.json"

RENDER_SCHEMA_VERSION = 1
CONDITIONS = (
    "both_readable",
    "feature_readable",
    "base_readable",
    "both_scrambled",
    "target_readable",
    "target_scrambled",
)

PILOT_PATH = WORD_LISTS / "composition_pilot.tsv"
CONFIRMATORY_PATH = WORD_LISTS / "composition_confirmatory.tsv"


def load_curated_items() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The pilot and confirmatory items, their check report and the dataset fingerprint.

    Raises when any check fails.
    """
    pilot = read_table(PILOT_PATH, "pilot")
    confirmatory = read_table(CONFIRMATORY_PATH, "confirmatory")
    items, report = validate_datasets(pilot, confirmatory)
    require_valid(report)
    return items, report, dataset_fingerprint(items)


def replicate_assignment(replicate: int) -> tuple[str, str]:
    """Return ``(feature_position, target_position)`` for one replicate.

    All four combinations of upper and lower occur over the four replicates.
    """
    assignment = {
        0: ("upper", "upper"),
        1: ("lower", "lower"),
        2: ("upper", "lower"),
        3: ("lower", "upper"),
    }
    if replicate not in assignment:
        raise ValueError(f"composition layout needs replicate 0-3, got {replicate}")
    return assignment[replicate]


def pair_centres(anchor: tuple[int, int], text_height: int,
                 panel_padding: int, gap: int) -> tuple[tuple[int, int],
                                                       tuple[int, int]]:
    """Two vertically separated centres around a shared anchor."""
    separation = text_height + 2 * panel_padding + gap
    upper = (int(anchor[0]), int(round(anchor[1] - separation / 2)))
    lower = (int(anchor[0]), int(round(anchor[1] + separation / 2)))
    return upper, lower


def _layout_fingerprint(layout: pd.DataFrame, dataset_sha: str) -> str:
    payload = layout.sort_values("slot", kind="mergesort").to_csv(
        index=False, lineterminator="\n")
    return hashlib.sha256(
        f"composition-layout-v{RENDER_SCHEMA_VERSION}|{dataset_sha}\n{payload}"
        .encode()).hexdigest()


def _box_panel_bounds(box: Box, pad: int) -> tuple[int, int, int, int]:
    return box.x - pad, box.y - pad, box.x + box.w + pad, box.y + box.h + pad


def _assert_pair_fits(upper: tuple[int, int], lower: tuple[int, int],
                      max_width: int, height: int, pad: int) -> None:
    wx, wy, side = crop_safe_window()
    for centre in (upper, lower):
        box = box_at_centre(centre, max_width, height)
        x0, y0, x1, y1 = _box_panel_bounds(box, pad)
        if not (x0 >= wx and x1 <= wx + side
                and y0 >= wy and y1 <= wy + side):
            raise ValueError(
                f"composition box including panel leaves crop-safe window: "
                f"{(x0, y0, x1, y1)} vs {(wx, wy, wx + side, wy + side)}")
    ub = box_at_centre(upper, max_width, height)
    lb = box_at_centre(lower, max_width, height)
    if _box_panel_bounds(ub, pad)[3] >= _box_panel_bounds(lb, pad)[1]:
        raise ValueError("composition word panels overlap")


def build_layout(items: pd.DataFrame) -> pd.DataFrame:
    """Compute one pair of word centres per slot, shared by every item."""
    words = sorted(set(items.feature) | set(items.base) | set(items.target))
    rows = []
    for slot in slot_table().itertuples(index=False):
        widths, heights = zip(*(
            text_extent(slot.font, slot.size, word) for word in words))
        max_width, height = max(widths), max(heights)
        pad = max(3, height // 6)
        gap = max(4, height // 4)
        separation = height + 2 * pad + gap
        footprint_half_h = math.ceil((separation + height + 2 * pad) / 2)
        footprint_half_w = math.ceil((max_width + 2 * pad) / 2)
        bg_path = BACKGROUNDS / f"bg_{int(slot.background_index):03d}.png"
        if not bg_path.exists():
            raise FileNotFoundError(
                f"missing background {bg_path}; run scripts/stimuli/backgrounds.py")
        with Image.open(bg_path) as background:
            anchor = legible_centre(
                background.convert("RGB"), slot.color, slot.size,
                footprint_half_w, footprint_half_h, slot.placement)
        upper, lower = pair_centres(anchor, height, pad, gap)
        _assert_pair_fits(upper, lower, max_width, height, pad)
        feature_position, target_position = replicate_assignment(
            int(slot.replicate))
        rows.append({
            **slot._asdict(),
            "anchor_x": anchor[0], "anchor_y": anchor[1],
            "upper_x": upper[0], "upper_y": upper[1],
            "lower_x": lower[0], "lower_y": lower[1],
            "feature_position": feature_position,
            "target_position": target_position,
            "max_text_width": max_width, "text_height": height,
            "panel_padding": pad, "word_gap": gap,
        })
    return pd.DataFrame(rows).sort_values("slot", kind="mergesort").reset_index(
        drop=True)


def load_or_create_layout(items: pd.DataFrame,
                          dataset_sha: str) -> tuple[pd.DataFrame, str]:
    """The recorded layout for this dataset, built and recorded when absent."""
    if LAYOUT_PATH.exists() and LAYOUT_META_PATH.exists():
        meta = json.loads(LAYOUT_META_PATH.read_text())
        if (meta.get("dataset_sha") != dataset_sha
                or meta.get("render_schema") != RENDER_SCHEMA_VERSION):
            raise RuntimeError(
                "composition layout belongs to a different dataset or render "
                "schema; delete composition/ to rebuild it")
        layout = pd.read_parquet(LAYOUT_PATH)
        got = _layout_fingerprint(layout, dataset_sha)
        if got != meta.get("layout_sha"):
            raise RuntimeError("composition layout fingerprint does not match metadata")
        return layout, got

    layout = build_layout(items)
    layout_sha = _layout_fingerprint(layout, dataset_sha)
    COMPOSITION.mkdir(parents=True, exist_ok=True)
    layout.to_parquet(LAYOUT_PATH, index=False)
    LAYOUT_META_PATH.write_text(json.dumps({
        "dataset_sha": dataset_sha,
        "layout_sha": layout_sha,
        "render_schema": RENDER_SCHEMA_VERSION,
    }, indent=2) + "\n")
    return layout, layout_sha


def _equation_dir(row, root: Path) -> Path:
    def value(name: str):
        return row[name] if isinstance(row, dict) else getattr(row, name)
    return root / str(value("split")) / str(value("family_id")) / str(
        value("question_id"))


def build_manifest(items: pd.DataFrame, layout: pd.DataFrame,
                   dataset_sha: str, layout_sha: str,
                   renders_root: Path = RENDERS_ROOT) -> pd.DataFrame:
    """Return the canonical expected manifest without rendering images."""
    rows = []
    layout_records = layout.to_dict("records")
    for item in items.itertuples(index=False):
        outdir = _equation_dir(item, renders_root)
        for slot in layout_records:
            feature_seed = _seed(
                dataset_sha, item.question_id, int(slot["slot"]), "feature")
            base_seed = _seed(
                dataset_sha, item.question_id, int(slot["slot"]), "base")
            target_seed = _seed(
                dataset_sha, item.question_id, int(slot["slot"]), "target")
            for condition_index, condition in enumerate(CONDITIONS):
                rows.append({
                    "dataset_sha": dataset_sha,
                    "layout_sha": layout_sha,
                    "render_schema": RENDER_SCHEMA_VERSION,
                    "split": item.split,
                    "family_id": item.family_id,
                    "equation_id": item.question_id,
                    "feature": item.feature,
                    "base": item.base,
                    "target": item.target,
                    "slot": int(slot["slot"]),
                    "appearance": int(slot["appearance"]),
                    "replicate": int(slot["replicate"]),
                    "background_index": int(slot["background_index"]),
                    "font": slot["font"], "size": slot["size"],
                    "color": slot["color"], "placement": slot["placement"],
                    "box": slot["box"],
                    "condition": condition,
                    "condition_index": condition_index,
                    "feature_position": slot["feature_position"],
                    "target_position": slot["target_position"],
                    "upper_x": int(slot["upper_x"]),
                    "upper_y": int(slot["upper_y"]),
                    "lower_x": int(slot["lower_x"]),
                    "lower_y": int(slot["lower_y"]),
                    "feature_seed": feature_seed,
                    "base_seed": base_seed,
                    "target_seed": target_seed,
                    "image_path": manifest_path(outdir / f"{int(slot['slot']):03d}_{condition}.png"),
                })
    frame = pd.DataFrame(rows)
    split_order = frame.split.map({"pilot": 0, "confirmatory": 1})
    frame.insert(0, "split_index", split_order)
    frame = (frame.sort_values(
        ["split_index", "family_id", "equation_id", "slot", "condition_index"],
        kind="mergesort").drop(columns="split_index").reset_index(drop=True))
    return frame


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """sha256 of every manifest column, as CSV in row order."""
    payload = manifest.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def _seed(dataset_sha: str, equation_id: str, slot: int, role: str) -> int:
    value = hashlib.sha256(
        f"{dataset_sha}|{equation_id}|{slot}|{role}".encode()).hexdigest()
    return int(value[:12], 16) % (2 ** 31)


def _centre(layout: dict, position: str) -> tuple[int, int]:
    return int(layout[f"{position}_x"]), int(layout[f"{position}_y"])


def _render_tokens(background: Image.Image, tokens: list[tuple[str, str, str, int]],
                   layout: dict) -> Image.Image:
    """Draw ``(text, position, mode, seed)`` tokens onto one background."""
    image = background
    for text, position, mode, seed in tokens:
        width, height = text_extent(layout["font"], layout["size"], text)
        box = box_at_centre(_centre(layout, position), width, height)
        kwargs = dict(color_group=layout["color"], box_style=layout["box"])
        if mode == "readable":
            image = render_overlay(image, text, box, layout["font"],
                                   layout["size"], **kwargs)
        elif mode == "scrambled":
            image = render_scramble(image, text, box, layout["font"],
                                    layout["size"], seed=seed, **kwargs)
        else:
            raise ValueError(f"unknown token mode {mode!r}")
    return image


def rendering_conditions(item) -> dict[str, tuple[tuple[str, str], ...]]:
    """The words each condition draws for one item, as ``(word, mode)`` pairs.

    Conditions follow ``CONDITIONS`` and words follow the drawing order of
    ``_render_job``. ``item`` needs ``feature``, ``base`` and ``target``.
    """
    feature, base, target = item["feature"], item["base"], item["target"]
    return {
        "both_readable": ((feature, "readable"), (base, "readable")),
        "feature_readable": ((feature, "readable"), (base, "scrambled")),
        "base_readable": ((feature, "scrambled"), (base, "readable")),
        "both_scrambled": ((feature, "scrambled"), (base, "scrambled")),
        "target_readable": ((target, "readable"),),
        "target_scrambled": ((target, "scrambled"),),
    }


def _render_job(job: dict) -> int:
    item, layout = job["item"], job["layout"]
    outdir = Path(job["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {condition: outdir / f"{int(layout['slot']):03d}_{condition}.png"
             for condition in CONDITIONS}
    needed = CONDITIONS if job["overwrite"] else tuple(
        condition for condition in CONDITIONS if not paths[condition].exists())
    if not needed:
        return 0

    feature_position = layout["feature_position"]
    base_position = "lower" if feature_position == "upper" else "upper"
    fseed = _seed(job["dataset_sha"], item["question_id"], layout["slot"],
                  "feature")
    bseed = _seed(job["dataset_sha"], item["question_id"], layout["slot"], "base")
    tseed = _seed(job["dataset_sha"], item["question_id"], layout["slot"],
                  "target")
    tokens = {
        "both_readable": [
            (item["feature"], feature_position, "readable", fseed),
            (item["base"], base_position, "readable", bseed)],
        "feature_readable": [
            (item["feature"], feature_position, "readable", fseed),
            (item["base"], base_position, "scrambled", bseed)],
        "base_readable": [
            (item["feature"], feature_position, "scrambled", fseed),
            (item["base"], base_position, "readable", bseed)],
        "both_scrambled": [
            (item["feature"], feature_position, "scrambled", fseed),
            (item["base"], base_position, "scrambled", bseed)],
        "target_readable": [
            (item["target"], layout["target_position"], "readable", tseed)],
        "target_scrambled": [
            (item["target"], layout["target_position"], "scrambled", tseed)],
    }
    bg_path = BACKGROUNDS / f"bg_{int(layout['background_index']):03d}.png"
    with Image.open(bg_path) as source:
        background = source.convert("RGB")
        for condition in needed:
            _render_tokens(background.copy(), tokens[condition], layout).save(
                paths[condition], "PNG")
    return len(needed)


def render_dataset(items: pd.DataFrame, layout: pd.DataFrame, dataset_sha: str,
                   *, workers: int = 12) -> pd.DataFrame:
    """Render all six conditions, skipping images already on disk."""
    RENDERS_ROOT.mkdir(parents=True, exist_ok=True)
    layout_sha = _layout_fingerprint(layout, dataset_sha)
    manifest = build_manifest(items, layout, dataset_sha, layout_sha)
    manifest_sha = manifest_fingerprint(manifest)
    if MANIFEST_META_PATH.exists():
        previous = json.loads(MANIFEST_META_PATH.read_text())
        expected = {
            "dataset_sha": dataset_sha,
            "layout_sha": layout_sha,
            "manifest_sha": manifest_sha,
            "render_schema": RENDER_SCHEMA_VERSION,
        }
        if any(previous.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "composition manifest belongs to a different dataset, layout, "
                "or render schema; delete composition/ to rebuild it")
    manifest.to_parquet(MANIFEST_PATH, index=False)

    def write_meta(complete: bool) -> None:
        MANIFEST_META_PATH.write_text(json.dumps({
            "dataset_sha": dataset_sha,
            "layout_sha": layout_sha,
            "manifest_sha": manifest_sha,
            "render_schema": RENDER_SCHEMA_VERSION,
            "rows": len(manifest),
            "render_complete": complete,
        }, indent=2) + "\n")

    write_meta(False)
    item_records = items.to_dict("records")
    layout_records = layout.to_dict("records")
    jobs = [{
        "item": item, "layout": slot,
        "outdir": str(_equation_dir(item, RENDERS_ROOT)),
        "dataset_sha": dataset_sha, "overwrite": False,
    } for item in item_records for slot in layout_records]

    written = processed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_render_job, jobs, chunksize=4)
        with tqdm(total=len(jobs) * len(CONDITIONS), desc="render composition",
                  unit="img", dynamic_ncols=True) as bar:
            for count in results:
                processed += 1
                written += count
                bar.update(len(CONDITIONS))
                bar.set_postfix(written=written,
                                reused=processed * len(CONDITIONS) - written)

    missing = manifest.loc[
        ~manifest.image_path.map(lambda value: local_path(value).exists()),
        "image_path"]
    if len(missing):
        raise RuntimeError(f"render completed with {len(missing)} missing images")
    write_meta(True)
    return manifest
