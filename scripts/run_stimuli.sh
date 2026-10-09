#!/usr/bin/env bash
# Stimuli: the 96 background photographs, the word, pair and audit tables,
# every word rendered on the 96 slots with its scrambled control, the class
# names and the class photographs, and the two-clue renders. Everything is
# written under the data root, VIT_SEMANTIC_DATA or data/ by default.
#
# Hardware: CPU. The OCR and CLIP steps of backgrounds.py and the text scan of
#   class_images.py use a GPU when one is present.
# Time: a few hours.
# Resumes: render.py, class_images.py and two_clue.py keep the images
#   already written, and class_images.py repeats its text scan. backgrounds.py,
#   vocabulary.py and class_names.py start over.
# Before: nothing. It needs the ImageNet-1k parquet export under imagenet-1k/
#   in the data root, the six font families of src/stimuli/typography.py and
#   NLTK WordNet.
#
# Run it once per data root. backgrounds.py overwrites the 96 photographs that
# every image is drawn on, and no cache fingerprint covers their pixels.
# render.py, class_images.py and two_clue.py write the manifests the
# caches are fingerprinted against, and a manifest that changes makes every
# cache built on the old one stale.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" scripts/stimuli/backgrounds.py
"$PY" scripts/stimuli/vocabulary.py
"$PY" scripts/stimuli/render.py
"$PY" scripts/stimuli/class_names.py
"$PY" scripts/stimuli/class_images.py
"$PY" scripts/stimuli/two_clue.py
