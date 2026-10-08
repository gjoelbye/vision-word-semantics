#!/usr/bin/env bash
# Analysis: one script per figure or table group writes its result tables to
# results/ from the caches, then the notebooks print the quoted numbers and
# write the figures to figures/ and the LaTeX tables to tables/. The executed
# notebooks, with their outputs, go to data/notebooks/, so the committed ones
# stay without outputs.
#
# Hardware: CPU.
# Time: about an hour.
# Resumes: no. Each script recomputes its results from the caches.
# Before: run_features.sh and run_finetune.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" scripts/analysis/figure1_neighbours.py
"$PY" scripts/analysis/figure3_language_alignment.py
"$PY" scripts/analysis/figure5_composition.py
"$PY" scripts/analysis/figure6_finetune.py
"$PY" scripts/analysis/table1_models.py
"$PY" scripts/analysis/table2_concepts.py
"$PY" -m jupyter nbconvert --to notebook --execute \
    --output-dir "${VIT_SEMANTIC_DATA:-data}/notebooks" \
    --ExecutePreprocessor.kernel_name=python3 notebooks/*.ipynb
