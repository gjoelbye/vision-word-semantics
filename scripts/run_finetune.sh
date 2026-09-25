#!/usr/bin/env bash
# Finetuning: the SPEECH-COCO shards and the list of photographs with text, the
# frozen caption targets of Qwen3-0.6B and w2v-BERT 2.0, the twenty
# contrastive finetunes, and the vocabulary caches of the finetuned models.
#
# Hardware: one GPU with 32 GB.
# Time: two to three days, most of it the twenty training runs.
# Resumes: caption_targets.py skips a finished cache and continues the speech
#   pass at its last finished shard. train.py skips a run that has its
#   provenance and continues an unfinished one from latest.pt. embed.py
#   continues each cache at its last recorded row. speech_coco.py starts over.
# Before: run_stimuli.sh, for the vocabulary manifest. It needs the SPEECH-COCO
#   parquet export under spokencoco_hf/data/ in the data root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" scripts/finetune/speech_coco.py
"$PY" scripts/finetune/caption_targets.py
"$PY" scripts/finetune/train.py --all
"$PY" scripts/features/embed.py --finetuned
