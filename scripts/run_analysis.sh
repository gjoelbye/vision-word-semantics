#!/usr/bin/env bash
# Analysis: the result tables behind every figure and table of the paper, one
# script per figure or table group, then the notebooks, executed in place,
# which print the quoted numbers and write the figures and LaTeX tables.
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
"$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 notebooks/*.ipynb
