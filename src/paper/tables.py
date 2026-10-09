"""One builder per generated LaTeX table in `tables/`.

The notebooks call these builders and write the result with `Table.write`. A
builder takes the result frames it prints, as `src.paper.results.read`
returns them, and returns a `Table` holding the printed cells as a
DataFrame and the LaTeX file. A column's largest value is bold and a value more
than two standard errors of the difference above the untrained ViT-B is
underlined; the encoder rows are references and take neither mark.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT, TABLES
from ..models import DEFAULT_SWEEP, PRETTY
from ..stats import binomial_se
from .latex import (BAND_NOTE, BASELINE, GROUPS, MARK_SIGMA, above_baseline,
                    band, band_colour, bare, cell, class_header, fit,
                    header, model_rows, pval, rank_text, reference_rows,
                    printed_frame, se_suffix, without_zero, zero)


@dataclass
class Table:
    """A generated table: its file name, its printed cells and its LaTeX source."""

    name: str
    frame: pd.DataFrame
    tex: str

    def write(self) -> Path:
        """Write `tables/<name>.tex` and return its path."""
        TABLES.mkdir(parents=True, exist_ok=True)
        path = TABLES / f"{self.name}.tex"
        path.write_text(self.tex)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        return path


def _fingerprint(*frames: pd.DataFrame, column: str = "manifest_sha") -> str:
    """The one fingerprint recorded in `column` of every frame."""
    values = set().union(*(set(frame[column]) for frame in frames))
    if len(values) != 1:
        raise RuntimeError(f"the inputs record {len(values)} values of {column}")
    return values.pop()


def _class_fingerprints(*frames: pd.DataFrame) -> tuple[str, str]:
    return (_fingerprint(*frames, column="word_manifest_sha"),
            _fingerprint(*frames, column="image_manifest_sha"))


def _table(name: str, lines: list[str], printed: list, columns: list[str],
           index_name: str = "Model") -> Table:
    return Table(name, printed_frame(printed, columns, index_name),
                 "\n".join(fit(lines)) + "\n")


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    rows = frame
    for column, value in where.items():
        rows = rows[rows[column].eq(value)]
    assert len(rows) == 1, (where, len(rows))
    return rows.iloc[0]


def _check_groups() -> None:
    listed = [k for group in GROUPS for k in group]
    if listed != list(DEFAULT_SWEEP):
        raise RuntimeError(f"GROUPS {listed} is not the sweep {DEFAULT_SWEEP}")


# Figure 1a.

# The models and the three queries of Figure 1a.
PANEL_MODELS = ("dinov3-h", "dinov3-7b", "webssl-dino-1b", "dinov2-g", "clip-b")
PANEL_QUERIES = (("germany", "Capital"), ("petrol", "Synonym"),
                 ("porsche", "Category"))
# Nearest words printed per query.
PANEL_DEPTH = 3


def _neighbour_cell(row: pd.Series) -> str:
    """A cosine and its word, bold where the word is a meaning partner or of the query's category."""
    word = str(row.neighbour)
    if bool(row.semantic_partner) or bool(row.same_category):
        word = f"\\textbf{{{word}}}"
    return f"{without_zero(f'{float(row.cosine):.2f}')} {word}"


def _neighbour_block(frame: pd.DataFrame, printed: list) -> list[str]:
    """PANEL_DEPTH lines per model, one column per query of PANEL_QUERIES."""
    models = PANEL_MODELS
    family = {model: next(i for i, group in enumerate(GROUPS) if model in group)
              for model in models}
    order = {f: i for i, f in enumerate(dict.fromkeys(family[m] for m in models))}
    lines = []
    for n, model in enumerate(models):
        if n and family[model] != family[models[n - 1]]:
            lines.append("\\addlinespace")
        for rank in range(1, PANEL_DEPTH + 1):
            cells = [_neighbour_cell(_one(frame, pool="vocabulary", model=model,
                                          query=query, rank=rank))
                     for query, _ in PANEL_QUERIES]
            label = PRETTY[model] if rank == 1 else ""
            printed.append((f"{PRETTY[model]} {rank}", cells))
            lines.append(band(order[family[model]]) + f"{label} & "
                         + " & ".join(cells) + " \\\\")
    return lines


def figure1a_neighbours(neighbours: pd.DataFrame) -> Table:
    """The three nearest words of three queries among the other 1,560, for five models."""
    heads = " & ".join(title for _, title in PANEL_QUERIES)
    rules = "".join(f"\\cmidrule(l{{2pt}}r{{2pt}}){{{j + 2}-{j + 2}}}"
                    for j in range(len(PANEL_QUERIES)))
    lines = [header("figure1_neighbours", ["neighbours"],
                    _fingerprint(neighbours)),
             "% The three nearest written words of each query, over the other "
             "1,560 words, names and pseudowords. Bold marks a meaning partner "
             "or a brand of the query's own industry." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}"
             "{@{\\extracolsep{\\fill}}" + "l" * (len(PANEL_QUERIES) + 1) + "@{}}",
             "\\toprule",
             f" & {heads} \\\\",
             rules,
             "Model & " + " & ".join(query for query, _ in PANEL_QUERIES)
             + " \\\\",
             "\\midrule"]
    printed = []
    lines += _neighbour_block(neighbours, printed)
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return _table("figure1a_neighbours", lines, printed,
                  [f"{title}: {query}" for query, title in PANEL_QUERIES],
                  index_name="Model and rank")


# Table 1.

# Table 1's columns: the two purity lists, then the top-10 relations; the last
# pools two relations over their summed pairs.
TABLE1_COLUMNS = (("category",), ("brand",), ("country_capital",), ("synonym",),
                  ("spelling_twin",), ("negation_base", "singular_plural"))


def table1_models(purity: pd.DataFrame, retrieval: pd.DataFrame,
                  encoder_lexical: pd.DataFrame) -> Table:
    """Neighbor purity on both lists and forward top-10 accuracy, with the encoders below."""
    _check_groups()
    forward = retrieval[retrieval.direction.eq("forward")]

    def value_and_se(model: str, j: int) -> tuple[float, float]:
        if j < 2:
            row = _one(purity, model=model, axis=TABLE1_COLUMNS[j][0])
            return float(row.purity), float(row.se)
        rows = forward[forward.model.eq(model)
                       & forward.relation.isin(TABLE1_COLUMNS[j])]
        assert len(rows) == len(TABLE1_COLUMNS[j]), (model, j, len(rows))
        hits, n = int(rows.hits10.sum()), int(rows.n.sum())
        return float(rows.hits10.sum() / rows.n.sum()), binomial_se(hits, n)

    table = {m: [value_and_se(m, j) for j in range(6)] for m in DEFAULT_SWEEP}
    chance = [float(purity[purity.axis.eq(axis)].chance.iloc[0])
              for axis in ("category", "brand")]
    chance += [float(forward.chance10.iloc[0])] * 4
    tops = [max(row[j][0] for row in table.values()) for j in range(6)]
    base = table[BASELINE]

    def cells(model: str) -> list[str]:
        return [cell(v, abs(v - tops[j]) < 1e-9,
                     model != BASELINE
                     and above_baseline(v, se, base[j][0], base[j][1]), se=se)
                for j, (v, se) in enumerate(table[model])]

    def encoder_cells(model: str) -> list[str]:
        out = []
        for j, relations in enumerate(TABLE1_COLUMNS):
            readout, direction = ("purity", "") if j < 2 else ("rate10", "forward")
            row = _one(encoder_lexical, model=model, readout=readout,
                       group="+".join(relations), direction=direction)
            if abs(float(row.chance) - chance[j]) > 1e-9:
                raise RuntimeError(f"{model} {relations}: chance {float(row.chance)} "
                                   f"is not the table's {chance[j]}")
            out.append(f"{float(row.estimate):.3f}{se_suffix(float(row.se))}")
        return out

    lines = [header("table1_models", ["purity", "retrieval", "encoder_lexical"],
                    _fingerprint(purity, retrieval, encoder_lexical)),
             f"% Underlining marks a value above the untrained ViT-B by more "
             f"than {MARK_SIGMA} standard errors of the difference; bold marks "
             "a column's largest value; the small number after a value is its "
             "standard error. The encoders under the models' rule are "
             "references and take no part in either." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
             "\\toprule",
             " & \\multicolumn{2}{c}{Neighbor purity} "
             "& \\multicolumn{4}{c}{Top-10 accuracy} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(l){4-7}",
             "Model & Nouns & Brands & Capital & Synonym & One-edit word "
             "& Plural or base \\\\",
             "\\midrule"]
    printed = []
    model_rows(lines, printed, DEFAULT_SWEEP, cells)
    reference_rows(lines, printed, encoder_cells, rule_before_speech=True)
    chance_cells = [bare(f"{v:.3f}") for v in chance]
    printed.append(("Chance", chance_cells))
    lines += ["\\midrule",
              "Chance & " + " & ".join(chance_cells) + " \\\\",
              "\\bottomrule",
              "\\end{tabular*}"]
    return _table("table1_models", lines, printed,
                  ["Nouns purity", "Brands purity", "Capital", "Synonym",
                   "One-edit word", "Plural or base"])


# Table 4.

# Relations of Table 4 and their column heads.
TABLE4_RELATIONS = (("country_capital", "Capital"),
                    ("synonym", "Synonym"),
                    ("antonym_synonym", "\\shortstack{Synonym\\\\(antonym\\\\list)}"),
                    ("antonym", "Antonym"),
                    ("negation", "\\shortstack{Synonym\\\\(negation\\\\list)}"),
                    ("spelling_twin", "\\shortstack{One-edit\\\\word}"),
                    ("singular_plural", "Plural"),
                    ("negation_base", "Base"))
# Type size of Table 4's standard errors.
TABLE4_SE_SIZE = "\\tiny"


def table4_lexical_retrieval(retrieval: pd.DataFrame,
                             encoder_lexical: pd.DataFrame) -> Table:
    """Top-10 accuracy for every relation, forward and reverse, one panel each."""
    _check_groups()
    relations = TABLE4_RELATIONS
    chosen = retrieval[retrieval.model.isin(DEFAULT_SWEEP)]
    assert len(chosen) == len(DEFAULT_SWEEP) * len(relations) * 2
    assert set(chosen.relation) == {r for r, _ in relations}
    assert chosen.chance10.nunique() == 1
    chance = float(chosen.chance10.iloc[0])
    indexed = chosen.set_index(["model", "relation", "direction"])

    lines = [header("table1_models", ["retrieval", "encoder_lexical"],
                    _fingerprint(retrieval, encoder_lexical)),
             f"% Underlining marks a value above the untrained ViT-B in the "
             f"same column of the same panel by more than {MARK_SIGMA} "
             "standard errors of the difference; bold marks a column's largest "
             "value; the small number after a value is its standard error. The "
             "encoders under the models' rule are references and take no part "
             "in either." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}"
             + "l" + "r" * len(relations) + "@{}}",
             "\\toprule",
             " & \\multicolumn{5}{c}{Meaning} & \\multicolumn{3}{c}{Spelling} \\\\",
             "\\cmidrule(lr){2-6}\\cmidrule(l){7-9}",
             "Model & " + " & ".join(h for _, h in relations) + " \\\\"]
    printed = []
    for label, direction in (("(a) Forward", "forward"),
                             ("(b) Reverse", "backward")):
        table = {m: [(float(indexed.loc[(m, r, direction)].hits10
                            / indexed.loc[(m, r, direction)].n),
                      float(indexed.loc[(m, r, direction)].rate10_se))
                     for r, _ in relations] for m in DEFAULT_SWEEP}
        tops = [max(row[j][0] for row in table.values())
                for j in range(len(relations))]
        base = table[BASELINE]

        def render(m: str) -> list[str]:
            return [cell(v, abs(v - tops[j]) < 1e-9,
                         m != BASELINE
                         and above_baseline(v, se, base[j][0], base[j][1]),
                         se=se, se_size=TABLE4_SE_SIZE)
                    for j, (v, se) in enumerate(table[m])]

        panel = []
        lines += ["\\midrule",
                  f"\\multicolumn{{{len(relations) + 1}}}{{l}}"
                  f"{{\\textbf{{{label}}}}} \\\\",
                  "\\addlinespace[2pt]"]
        model_rows(lines, panel, DEFAULT_SWEEP, render)
        reference_rows(lines, panel, lambda m: [
            f"{float(row.estimate):.3f}"
            + se_suffix(float(row.se), size=TABLE4_SE_SIZE)
            for row in (_one(encoder_lexical, model=m, readout="rate10",
                             group=r, direction=direction)
                        for r, _ in relations)])
        chance_cells = [bare(f"{chance:.3f}", size=TABLE4_SE_SIZE)
                        for _ in relations]
        panel.append(("Chance", chance_cells))
        lines += ["\\addlinespace",
                  "Chance & " + " & ".join(chance_cells) + " \\\\"]
        printed += [(f"{label[4:]}: {row}", cells) for row, cells in panel]
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return _table("table4_lexical_retrieval", lines, printed,
                  ["Capital", "Synonym", "Synonym (antonym list)", "Antonym",
                   "Synonym (negation list)", "One-edit word", "Plural", "Base"],
                  index_name="Direction: model")


# Table 5.

def table5_items(items: pd.DataFrame, retrieval: pd.DataFrame,
                 encoder_lexical: pd.DataFrame) -> Table:
    """The three comparisons within an item and the concrete-abstract contrast.

    A star marks a count where the synonym is ahead and the unadjusted sign or
    Fisher test gives p below 0.05. Only the two similarity columns are
    underlined, because the counts carry no standard error.
    """
    _check_groups()
    head = items[items.record_type.eq("head_to_head")]
    contrast = items[items.record_type.eq("class_contrast")]
    # Chance of a class count: how many of 75 anchors are expected to have their
    # synonym in the top 10 in either direction.
    within = float(retrieval.chance10.iloc[0])
    class_chance = 75 * (1 - (1 - within) ** 2)
    blocks = (("synonym_vs_twin", 150), ("synonym_vs_antonym", 50),
              ("synonym_vs_base", 35))

    def starred(value: int, ahead: bool, p: float) -> str:
        return f"{value}$^{{*}}$" if ahead and p < 0.05 else f"{value}"

    def z(model: str, column: str) -> tuple[float, float]:
        row = _one(head, model=model, block="synonym_vs_twin")
        return float(row[column]), float(row[f"{column}_se"])

    synonym_top = max(z(m, "kept_above_baseline")[0] for m in DEFAULT_SWEEP)
    synonym_base, synonym_base_se = z(BASELINE, "kept_above_baseline")
    twin_base, twin_base_se = z(BASELINE, "other_above_baseline")
    top = max(max(int(_one(contrast, model=m).concrete_found),
                  int(_one(contrast, model=m).abstract_found))
              for m in DEFAULT_SWEEP)

    def reference(model: str) -> list[str]:
        cells = []
        for j, (block, _) in enumerate(blocks):
            wins = _one(encoder_lexical, model=model, readout="wins",
                        group=block, direction="")
            cells.append(f"{int(wins.estimate)}")
            if j == 0:
                for readout in ("other_above_baseline", "kept_above_baseline"):
                    row = _one(encoder_lexical, model=model, readout=readout,
                               group=block, direction="")
                    cells.append(f"{float(row.estimate):.2f}"
                                 + se_suffix(float(row.se), 2))
        for cls in ("concrete", "abstract"):
            row = _one(encoder_lexical, model=model, readout="class_contrast",
                       group=cls, direction="either")
            cells.append(f"{int(row.estimate)}")
        return cells

    lines = [header("table1_models", ["items", "retrieval", "encoder_lexical"],
                    _fingerprint(items, retrieval, encoder_lexical)),
             "% A star marks a count where the synonym is ahead and "
             "the test says so, unadjusted. Underlining marks a value above "
             f"the untrained ViT-B by more than {MARK_SIGMA} standard errors "
             "of the difference, which the two similarity columns carry a "
             "standard error for and the counts do not. The small number after a similarity "
             "is its standard error over items. The encoders under the models' "
             "rule are references: they carry no test and no mark." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}"
             "{@{\\extracolsep{\\fill}}lrrrrrrr@{}}",
             "\\toprule",
             " & \\multicolumn{3}{c}{\\shortstack{Synonym against\\\\one-edit word}} "
             "& \\multicolumn{1}{c}{\\shortstack{Synonym\\\\against\\\\antonym}} "
             "& \\multicolumn{1}{c}{\\shortstack{Synonym\\\\against\\\\base}} "
             "& \\multicolumn{2}{c}{\\shortstack{Top-10 hits\\\\for synonyms}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}"
             "\\cmidrule(lr){6-6}\\cmidrule(l){7-8}",
             "Model & \\shortstack{Count\\\\of 150} & \\shortstack{One-edit\\\\word} & Synonym "
             "& \\shortstack{Count\\\\of 50} & \\shortstack{Count\\\\of 35} "
             "& Concrete & Abstract \\\\",
             "\\midrule"]
    def render(model: str) -> list[str]:
        cells = []
        for j, (block, n) in enumerate(blocks):
            row = _one(head, model=model, block=block)
            cells.append(starred(int(row.wins), int(row.wins) > n / 2,
                                 float(row.sign_p)))
            if j == 0:
                far, far_se = z(model, "other_above_baseline")
                cells.append(cell(far, False,
                                  model != BASELINE
                                  and above_baseline(far, far_se, twin_base,
                                                     twin_base_se),
                                  se=far_se, digits=2))
                near, near_se = z(model, "kept_above_baseline")
                cells.append(cell(near, abs(near - synonym_top) < 1e-9,
                                  model != BASELINE
                                  and above_baseline(near, near_se,
                                                     synonym_base,
                                                     synonym_base_se),
                                  se=near_se, digits=2))
        cls = _one(contrast, model=model)
        concrete = starred(int(cls.concrete_found), True, float(cls.fisher_p))
        for value, text in ((int(cls.concrete_found), concrete),
                            (int(cls.abstract_found),
                             str(int(cls.abstract_found)))):
            if value == top:
                text = text.replace(str(value), f"\\textbf{{{value}}}", 1)
            cells.append(text)
        return cells

    printed = []
    model_rows(lines, printed, DEFAULT_SWEEP, render)
    reference_rows(lines, printed, reference)
    chance_cells = ["75", bare("0.00", 2), bare("0.00", 2), "25", "17.5",
                    f"{class_chance:.2f}", f"{class_chance:.2f}"]
    printed.append(("Chance", chance_cells))
    lines += ["\\addlinespace",
              "Chance & " + " & ".join(chance_cells) + " \\\\",
              "\\bottomrule", "\\end{tabular*}"]
    return _table("table5_items", lines, printed,
                  ["Synonym vs one-edit: count of 150", "One-edit word z",
                   "Synonym z", "Synonym vs antonym: count of 50",
                   "Synonym vs base: count of 35", "Concrete top-10 hits",
                   "Abstract top-10 hits"])


# Table 6.

def table6_category(purity: pd.DataFrame, latent_categories: pd.DataFrame,
                    encoder_lexical: pd.DataFrame) -> Table:
    """Neighbor purity and nearest-centroid accuracy on both lists, each with its p.

    Every p comes from a null that shuffles categories within word-length
    terciles.
    """
    _check_groups()

    def columns(model: str):
        pure = [_one(purity, model=model, axis=axis) for axis in ("category", "brand")]
        held = [_one(latent_categories, model=model, dataset=d)
                for d in ("common_noun", "brand")]
        return ([(float(r.purity), float(r.se), float(r.p), float(r.chance))
                 for r in pure]
                + [(float(r.full_classification_accuracy),
                    float(r.full_classification_accuracy_se),
                    float(r.length_stratified_full_classification_accuracy_p),
                    float(r.classification_chance)) for r in held])

    table = {model: columns(model) for model in DEFAULT_SWEEP}
    chance = [column[3] for column in table[DEFAULT_SWEEP[0]]]
    tops = [max(table[m][j][0] for m in DEFAULT_SWEEP) for j in range(4)]
    base = table[BASELINE]
    reference_where = (("purity", "category"), ("purity", "brand"),
                       ("classification", "common_noun"),
                       ("classification", "brand"))

    def reference(model: str) -> list[str]:
        cells = []
        for readout, group in reference_where:
            row = _one(encoder_lexical, model=model, readout=readout,
                       group=group, direction="")
            cells += [f"{float(row.estimate):.3f}" + se_suffix(float(row.se)), "-"]
        return cells

    lines = [header("table1_models",
                    ["purity", "latent_categories", "encoder_lexical"],
                    _fingerprint(purity, latent_categories, encoder_lexical)),
             "% Every p holds word length fixed: categories are shuffled within "
             "length tercile, terciles of word length within the list. The "
             "small number after a value is its standard error. Underlining "
             f"marks a value above the untrained ViT-B by more than "
             f"{MARK_SIGMA} standard errors of the difference; bold marks a "
             "column's largest value. The encoders under the models' rule are "
             "references: they set no bold and carry no mark and no p."
             + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}"
             "{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}",
             "\\toprule",
             " & \\multicolumn{4}{c}{Neighbor purity} "
             "& \\multicolumn{4}{c}{Nearest-centroid accuracy} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
             " & \\multicolumn{2}{c}{Nouns} & \\multicolumn{2}{c}{Brands} "
             "& \\multicolumn{2}{c}{Nouns} & \\multicolumn{2}{c}{Brands} "
             "\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"
             "\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
             "Model & Value & $p$ & Value & $p$ "
             "& Value & $p$ & Value & $p$ \\\\",
             "\\midrule"]
    def render(model: str) -> list[str]:
        cells = []
        for j, (value, se, p, _) in enumerate(table[model]):
            cells.append(cell(value, abs(value - tops[j]) < 1e-9,
                              model != BASELINE
                              and above_baseline(value, se, base[j][0],
                                                 base[j][1]),
                              se=se))
            cells.append(f"{p:.4f}")
        return cells

    printed = []
    model_rows(lines, printed, DEFAULT_SWEEP, render)
    reference_rows(lines, printed, reference)
    chance_cells = [text for c in chance for text in (bare(f"{c:.3f}"), "-")]
    printed.append(("Chance", chance_cells))
    lines += ["\\addlinespace",
              "Chance & " + " & ".join(f"{bare(f'{c:.3f}')} & -" for c in chance)
              + " \\\\",
              "\\bottomrule", "\\end{tabular*}"]
    return _table("table6_category", lines, printed,
                  ["Nouns purity", "Nouns purity p", "Brands purity",
                   "Brands purity p", "Nouns accuracy", "Nouns accuracy p",
                   "Brands accuracy", "Brands accuracy p"])


# Table 7.

# Type size of Table 7's standard errors.
TABLE7_SE_SIZE = "\\tiny"


def _positions(depth: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list, list, float]:
    """Depth rows after the input position, the positions, their mean relative depths and widest range."""
    depth = depth[depth.layer_index.ne(0)].dropna(subset=columns)
    positions = sorted(depth.layer_index.unique())
    missing = [(m, i) for m in DEFAULT_SWEEP for i in positions
               if len(depth[depth.model.eq(m) & depth.layer_index.eq(i)]) != 1]
    if missing:
        raise RuntimeError(f"the depth table is missing {missing[:4]}")
    labels = [depth.loc[depth.layer_index.eq(i), "relative_depth"].mean()
              for i in positions]
    spread = max(depth.loc[depth.layer_index.eq(i), "relative_depth"].max()
                 - depth.loc[depth.layer_index.eq(i), "relative_depth"].min()
                 for i in positions)
    return depth, positions, labels, spread


def _depth_block(lines: list[str], printed: list, tag: str, title: str,
                 positions: list, labels: list, render) -> None:
    """Append one block of a depth table: its title, a column per depth and a row per model."""
    lines += [f"\\textbf{{{tag}}} {title}\\par\\smallskip",
              "\\begin{tabular*}{\\textwidth}"
              "{@{\\extracolsep{\\fill}}l" + "r" * len(positions) + "@{}}",
              "\\toprule",
              f" & \\multicolumn{{{len(positions)}}}{{c}}{{Relative depth}}"
              " \\\\",
              f"\\cmidrule(l){{2-{len(positions) + 1}}}",
              "Model & " + " & ".join(f"{d:.2f}" for d in labels) + " \\\\",
              "\\midrule"]
    block = []
    model_rows(lines, block, DEFAULT_SWEEP, render)
    printed += [(f"{tag} {label}", cells) for label, cells in block]


def table7_lexical_depth(depth: pd.DataFrame, purity: pd.DataFrame,
                         retrieval: pd.DataFrame) -> Table:
    """Noun purity and capital top-10 accuracy at every cached depth after the input.

    Bold marks each model's largest value across depths where it is above
    chance; underlining compares with the untrained ViT-B at the same depth.
    """
    _check_groups()
    depth, positions, labels, spread = _positions(depth, ["purity", "retrieval10"])

    def value(model: str, position: int, column: str) -> float:
        return float(_one(depth, model=model, layer_index=position)[column])

    nouns = purity[purity.axis.eq("category")]
    assert nouns.chance.nunique() == 1
    capitals = retrieval[retrieval.relation.eq("country_capital")
                         & retrieval.direction.eq("forward")]
    assert capitals.chance10.nunique() == 1
    blocks = (("(a)", "Noun neighbor purity.", "purity",
               float(nouns.chance.iloc[0])),
              ("(b)", "Capital top-10 accuracy.", "retrieval10",
               float(capitals.chance10.iloc[0])))

    # The final depth must equal the final-layer tables, value and standard error.
    final = positions[-1]
    for model in DEFAULT_SWEEP:
        head = _one(nouns, model=model)
        assert abs(value(model, final, "purity") - float(head.purity)) < 1e-9
        assert abs(value(model, final, "purity_se") - float(head.se)) < 1e-9
        head = _one(capitals, model=model)
        assert abs(value(model, final, "retrieval10") - float(head.rate10)) < 1e-9
        assert abs(value(model, final, "retrieval10_se")
                   - float(head.rate10_se)) < 1e-9

    lines = [header("table1_models", ["depth", "purity", "retrieval"],
                    _fingerprint(depth, purity, retrieval)),
             f"% Columns are the cached hidden states after the input "
             f"position, labeled by the mean relative depth of each position; "
             f"the widest range of a position across models is {spread:.3f}. "
             f"Underlining marks a value above the untrained ViT-B at the "
             f"same depth by more than {MARK_SIGMA} standard errors of the "
             f"difference; bold marks each model's largest value across the "
             f"depths, where that value is above chance; the small number "
             f"after a value is its standard error." + BAND_NOTE,
             band_colour()]
    printed = []
    for n, (tag, title, column, chance) in enumerate(blocks):
        if n:
            lines.append("\\par\\medskip")

        def render(model: str) -> list[str]:
            values = [value(model, i, column) for i in positions]
            best = max(values)
            return [cell(v, v == best > chance,
                         model != BASELINE
                         and above_baseline(
                             v, value(model, i, f"{column}_se"),
                             value(BASELINE, i, column),
                             value(BASELINE, i, f"{column}_se")),
                         se=value(model, i, f"{column}_se"),
                         se_size=TABLE7_SE_SIZE, strip_zero=True)
                    for i, v in zip(positions, values)]

        _depth_block(lines, printed, tag, title, positions, labels, render)
        chance_cells = [bare(without_zero(f"{chance:.3f}"), size=TABLE7_SE_SIZE)
                        for _ in positions]
        printed.append((f"{tag} Chance", chance_cells))
        lines += ["\\midrule",
                  "Chance & " + " & ".join(chance_cells) + " \\\\",
                  "\\bottomrule", "\\end{tabular*}"]
    return _table("table7_lexical_depth", lines, printed,
                  [f"{d:.2f}" for d in labels], index_name="Block and model")


# Tables 2, 8, 9 and 10.

def _chance(retr: pd.DataFrame, subset: str, direction: str,
            column: str) -> float:
    """A chance level, asserted to be the same for every model."""
    rows = retr[retr.subset.eq(subset) & retr.direction.eq(direction)]
    assert rows[column].nunique() == 1, (subset, direction, column)
    return float(rows[column].iloc[0])


def _class_models(frame: pd.DataFrame) -> list[str]:
    have = [m for group in GROUPS for m in group if m in set(frame.model)]
    if BASELINE not in have:
        raise RuntimeError("the class-name results have no untrained ViT-B")
    return have


def table2_class_names(class_alignment: pd.DataFrame,
                       class_retrieval: pd.DataFrame) -> Table:
    """Matched cosine similarity and retrieval in both directions, on all photographs.

    A median rank has no standard error and no chance level, so it takes no
    underline, and bold marks its smallest value.
    """
    align, retr = class_alignment, class_retrieval
    have = _class_models(align)
    full = align[align.subset.eq("all")].set_index("model")
    directions = ("word_to_class", "class_to_word")
    rows, errors = {}, {}
    for model in have:
        values = [float(full.alignment[model])]
        se = {0: float(full.se[model])}
        for j, direction in zip((1, 3), directions):
            row = _one(retr, model=model, subset="all", direction=direction)
            values += [float(row.rate10), float(row.median_rank)]
            se[j] = float(row.rate10_se)
        rows[model], errors[model] = values, se
    ranks = (2, 4)
    chance = [float(full.null_mean.loc[have].mean())]
    for direction in directions:
        chance += [_chance(retr, "all", direction, "chance10"), float("nan")]
    tops = [max(r[j] for r in rows.values()) for j in range(len(chance))]
    lows = [min(r[j] for r in rows.values()) for j in range(len(chance))]
    base, base_se = rows[BASELINE], errors[BASELINE]

    def render(model: str) -> list[str]:
        v, se = rows[model], errors[model]
        cells = []
        for j in range(len(chance)):
            if j in ranks:
                text = rank_text(v[j])
                cells.append(f"\\textbf{{{text}}}"
                             if abs(v[j] - lows[j]) < 1e-9 else text)
            else:
                cells.append(cell(v[j], abs(v[j] - tops[j]) < 1e-9,
                                  model != BASELINE
                                  and above_baseline(v[j], se[j], base[j],
                                                     base_se[j]),
                                  se=se[j]))
        return cells

    lines = [class_header("table2_class_names",
                          ["class_alignment", "class_retrieval"],
                          *_class_fingerprints(align, retr)),
             f"% Underlining marks a value above the untrained ViT-B by more "
             f"than {MARK_SIGMA} standard errors of the difference; bold marks "
             "a column's largest value, or its smallest median rank; the small "
             "number after a value is its standard error, and a median rank "
             "has none." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}"
             "{@{\\extracolsep{\\fill}}lrrrrr@{}}",
             "\\toprule",
             " & & \\multicolumn{2}{c}{Word to class} "
             "& \\multicolumn{2}{c}{Class to word} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
             "Model & Matched cosine similarity "
             "& \\shortstack{Top-10\\\\accuracy} "
             "& \\shortstack{Median\\\\rank} "
             "& \\shortstack{Top-10\\\\accuracy} "
             "& \\shortstack{Median\\\\rank} \\\\",
             "\\midrule"]
    printed = []
    model_rows(lines, printed, have, render)
    chance_cells = ["" if c != c else bare(zero(c) if j == 0 else f"{c:.3f}")
                    for j, c in enumerate(chance)]
    printed.append(("Chance", chance_cells))
    lines += ["\\midrule",
              "Chance & " + " & ".join(chance_cells) + " \\\\",
              "\\bottomrule", "\\end{tabular*}"]
    return _table("table2_class_names", lines, printed,
                  ["Matched cosine similarity", "Word to class top-10",
                   "Word to class median rank", "Class to word top-10",
                   "Class to word median rank"])


def table8_class_alignment(class_alignment: pd.DataFrame) -> Table:
    """Matched cosine similarity with its standard error, null mean and p, and on text-free photographs."""
    align = class_alignment
    have = _class_models(align)
    full = align[align.subset.eq("all")].set_index("model")
    clean = align[align.subset.eq("text_free")].set_index("model")
    chance = [float(full.null_mean.loc[have].mean()),
              float(clean.null_mean.loc[have].mean())]
    tops = [max(float(full.alignment[m]) for m in have),
            max(float(clean.alignment[m]) for m in have)]
    base = [(float(full.alignment[BASELINE]), float(full.se[BASELINE])),
            (float(clean.alignment[BASELINE]), float(clean.se[BASELINE]))]

    def render(model: str) -> list[str]:
        cells = []
        for j, x in enumerate((full.loc[model], clean.loc[model])):
            mean, se = float(x.alignment), float(x.se)
            cells.append(cell(mean, abs(mean - tops[j]) < 1e-9,
                              model != BASELINE
                              and above_baseline(mean, se, *base[j]), se=se))
            if j == 0:
                cells += [zero(float(x.null_mean), 4), pval(float(x.p))]
        return cells

    n_clean = int(clean.n_words.iloc[0])
    lines = [class_header("table2_class_names", ["class_alignment"],
                          *_class_fingerprints(align)),
             "% One row per model. The small number after a value is its "
             "standard error over class names, the null mean is the mean matched cosine similarity "
             "over permuted pairings, and the column for class images with no "
             f"detected text covers {n_clean} class names. Underlining marks a "
             f"value in either of those two columns above the untrained ViT-B "
             f"by more than {MARK_SIGMA} standard errors of the difference; "
             "bold marks a column's largest value." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}"
             "{@{\\extracolsep{\\fill}}lrrrr@{}}",
             "\\toprule",
             " & \\multicolumn{3}{c}{Matched cosine similarity} & \\\\",
             "\\cmidrule(lr){2-4}",
             "Model & All images & Null mean & $p$ "
             "& \\shortstack{No detected\\\\text} \\\\",
             "\\midrule"]
    printed = []
    model_rows(lines, printed, have, render)
    chance_cells = [bare(zero(chance[0])), "", "", bare(zero(chance[1]))]
    printed.append(("Chance", chance_cells))
    lines += ["\\midrule",
              f"Chance & {chance_cells[0]} & & & {chance_cells[3]} \\\\",
              "\\bottomrule", "\\end{tabular*}"]
    return _table("table8_class_alignment", lines, printed,
                  ["All images", "Null mean", "p", "No detected text"])


def table9_class_retrieval(class_retrieval: pd.DataFrame) -> Table:
    """Top-10 accuracy and median rank in both directions, and top-10 accuracy on text-free photographs."""
    retr = class_retrieval
    have = _class_models(retr)
    directions = ("word_to_class", "class_to_word")
    per = 3
    values = {}
    for m in have:
        cells = []
        for d in directions:
            a = _one(retr, model=m, subset="all", direction=d)
            t = _one(retr, model=m, subset="text_free", direction=d)
            cells += [(float(a.rate10), float(a.rate10_se)),
                      (float(a.median_rank), None),
                      (float(t.rate10), float(t.rate10_se))]
        values[m] = cells
    chance = []
    for d in directions:
        chance += [_chance(retr, "all", d, "chance10"), float("nan"),
                   _chance(retr, "text_free", d, "chance10")]
    tops = [max(v[j][0] for v in values.values()) for j in range(len(chance))]
    lows = [min(v[j][0] for v in values.values()) for j in range(len(chance))]
    base = values[BASELINE]

    def render(model: str) -> list[str]:
        out = []
        for j, (v, se) in enumerate(values[model]):
            if j % per == 1:
                text = rank_text(v)
                out.append(f"\\textbf{{{text}}}" if abs(v - lows[j]) < 1e-9 else text)
            else:
                out.append(cell(v, abs(v - tops[j]) < 1e-9,
                                model != BASELINE
                                and above_baseline(v, se, base[j][0], base[j][1]),
                                se=se))
        return out

    def candidates(d: str) -> str:
        n = retr[retr.subset.eq("all") & retr.direction.eq(d)].n_candidates
        assert n.nunique() == 1
        return f"{int(n.iloc[0]):,}"

    heads = ("\\shortstack{Top-10\\\\accuracy} "
             "& \\shortstack{Median\\\\rank} & \\shortstack{No detected\\\\text}")
    lines = [class_header("table2_class_names", ["class_retrieval"],
                          *_class_fingerprints(retr)),
             "% One row per model, three columns per retrieval. Ties count for "
             "the correct candidate; the small number after a share is its "
             "standard error. Underlining marks a share above the untrained "
             f"ViT-B by more than {MARK_SIGMA} standard errors of the "
             "difference; bold marks a column's largest share and its smallest "
             "median rank." + BAND_NOTE,
             band_colour(),
             "\\begin{tabular*}{\\textwidth}"
             "{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
             "\\toprule",
             f" & \\multicolumn{{3}}{{c}}{{Word-to-class retrieval, {candidates('word_to_class')} classes}}"
             f" & \\multicolumn{{3}}{{c}}{{Class-to-word retrieval, {candidates('class_to_word')} class names}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
             f"Model & {heads} & {heads} \\\\",
             "\\midrule"]
    printed = []
    model_rows(lines, printed, have, render)
    chance_cells = ["" if c != c else bare(f"{c:.3f}") for c in chance]
    printed.append(("Chance", chance_cells))
    lines += ["\\midrule",
              "Chance & " + " & ".join(chance_cells) + " \\\\",
              "\\bottomrule", "\\end{tabular*}"]
    return _table("table9_class_retrieval", lines, printed,
                  [f"{d} {c}" for d in ("Word to class", "Class to word")
                   for c in ("top-10", "median rank", "top-10 no detected text")])


def table10_class_depth(class_depth: pd.DataFrame) -> Table:
    """Matched cosine similarity and word-to-class top-10 accuracy at every cached depth after the input.

    Bold marks each model's largest value across depths. Nothing is
    underlined, because the depth profile carries no standard errors.
    """
    _check_groups()
    depth, positions, labels, spread = _positions(class_depth, ["alignment"])

    def value(model: str, position: int, column: str) -> float:
        return float(_one(depth, model=model, layer_index=position)[column])

    blocks = (("(a)", "Matched cosine similarity.", "alignment", 0.0, False),
              ("(b)", "Word-to-class top-10 accuracy.",
               "word_to_class_rate10",
               float(depth.word_to_class_chance10.iloc[0]), True))
    lines = [class_header("table2_class_names", ["class_depth"],
                          *_class_fingerprints(depth)),
             f"% Columns are the cached hidden states after the input layer, "
             f"labeled by the mean relative depth of each position; the widest "
             f"range of a position across models is {spread:.3f}. This table "
             f"prints no standard errors, so no value is underlined; bold "
             f"marks each model's largest value across the depths."
             + BAND_NOTE,
             band_colour()]
    printed = []
    for n, (tag, title, column, chance, chance_row) in enumerate(blocks):
        if n:
            lines.append("\\par\\medskip")

        def render(model: str) -> list[str]:
            values = [value(model, i, column) for i in positions]
            best = max(values)
            return [cell(v, v == best, False) for v in values]

        _depth_block(lines, printed, tag, title, positions, labels, render)
        if chance_row:
            chance_cells = [f"{chance:.3f}" for _ in positions]
            printed.append((f"{tag} Chance", chance_cells))
            lines += ["\\midrule",
                      "Chance & " + " & ".join(chance_cells) + " \\\\"]
        lines += ["\\bottomrule", "\\end{tabular*}"]
    return _table("table10_class_depth", lines, printed,
                  [f"{d:.2f}" for d in labels], index_name="Block and model")
