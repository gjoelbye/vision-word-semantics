#!/usr/bin/env bash
# Features: the class-token caches of the 14 image models for the vocabulary
# (ten depths), the class names and class photographs (ten depths) and the
# two-clue renders (final layer); the word vectors of the three Qwen3 text
# encoders and of w2v-BERT 2.0; and the word-by-word similarity matrices of
# the 14 models and 4 encoders.
#
# Hardware: one GPU with 32 GB.
# Time: one to two days, most of it the vocabulary caches of the largest models.
# Resumes: an unfinished embedding cache continues at its last recorded row,
#   and a finished cache or word-vector file is skipped. similarity.py
#   recomputes every matrix.
# Before: run_stimuli.sh. word_speech.py needs phonemizer and the espeak-ng
#   binary.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" scripts/features/embed.py
"$PY" scripts/features/embed.py --manifest concepts
"$PY" scripts/features/embed.py --manifest two_clue
"$PY" scripts/features/word_text.py
"$PY" scripts/features/word_speech.py
"$PY" scripts/features/similarity.py
