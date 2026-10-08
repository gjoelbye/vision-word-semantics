# Data

This folder is the data root. It holds the two downloaded datasets and
everything the stimulus, feature and finetuning stages generate. Git tracks
only this file and `.gitkeep`. Set `VIT_SEMANTIC_DATA` to use another
directory; the code reads every path below relative to it.

The analysis outputs the paper reads are small and committed elsewhere: result
tables in `results/`, figures in `figures/` and LaTeX tables in `tables/`, all
in the repository root.

## Downloads

The ImageNet-1k validation images and SPEECH-COCO, which holds the COCO images
with written and spoken captions, take about 166 GB together.

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
    --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 \
    --include "data/validation-*" --include classes.py \
    --local-dir data/imagenet-1k

hf download mteb/SpeechCoco --repo-type dataset \
    --revision ba337542de8f4ee66cfc285b5eeb69c326dc1576 \
    --include "data/*" --local-dir data/spokencoco_hf
```

With `VIT_SEMANTIC_DATA` set, change the `--local-dir` paths to match.

| Folder | Content | Size |
|---|---|---|
| `imagenet-1k/` | The 14 validation shards and `classes.py` | 6.7 GB |
| `spokencoco_hf/` | The SPEECH-COCO parquet export | 159 GB |

The other requirements are not files in this folder: NLTK WordNet, espeak-ng
and the fonts listed in the main README.

## What each stage writes

Sizes are those of the recorded run, counting only the files the current code
writes. Together they come to about 420 GB, and with the downloads to about
590 GB, inside the 750 GB of free disk the main README asks for.

| Folder | Written by | Content | Size |
|---|---|---|---|
| `backgrounds/` | `scripts/stimuli/backgrounds.py` | The 96 background photographs and `backgrounds.parquet` | 13 MB |
| `lexicon/` | `scripts/stimuli/wordlists.py`, `render.py`, `class_names.py` | The word, pair and audit tables, the vocabulary manifest and text centres, and the class-name tables | 1.7 MB |
| `renders/` | `scripts/stimuli/render.py`, `class_images.py` | Each vocabulary word and class name on the 96 slots, and its scrambled control | 46 GB |
| `concepts/` | `scripts/stimuli/class_images.py` | The class-name manifest, the ImageNet validation photographs cropped to 256 pixels, their list and their text scan | 6.1 GB |
| `composition/` | `scripts/stimuli/two_clue_stimuli.py` | The two-clue layout, manifest and renders | 11 GB |
| `embeddings/` | `scripts/features/embed.py` | Class-token caches of the 14 image models at ten depths for the vocabulary | 104 GB |
| `embeddings/concepts/` | `scripts/features/embed.py --manifest concepts` | The same for the class-name renders and the class photographs | 31 GB |
| `composition/embeddings/` | `scripts/features/embed.py --manifest two_clue` | Final-layer caches of the two-clue renders | 2.9 GB |
| `word_text/` | `scripts/features/word_text.py` | Word vectors of the three Qwen3 text encoders | 4.7 GB |
| `word_speech/` | `scripts/features/word_speech.py` | Word vectors of w2v-BERT 2.0 from synthesized speech | 2.5 GB |
| `similarity/` | `scripts/features/similarity.py` | Word-by-word cosine matrices of the 14 models and 4 encoders, and `index.parquet` | 175 MB |
| `spokencoco-wds-288/` | `scripts/finetune/speech_coco.py` | SPEECH-COCO as webdataset shards, short side 288 | 36 GB |
| `corpus/` | `scripts/finetune/speech_coco.py` | The `speech-coco_` manifest, provenance, text scan and list of excluded photographs | 44 MB |
| `captions/speech-coco/` | `scripts/finetune/caption_targets.py` | Frozen caption targets of Qwen3-0.6B and w2v-BERT 2.0 | 2.5 GB |
| `checkpoints/` | `scripts/finetune/train.py` | The 20 finetunes, each with `best.slim.pt` and `provenance.json`; `latest.pt` holds a run in progress and is deleted when it ends | 39 GB, plus up to 14 GB while a run is in progress |
| `embeddings/<model>-scocoft*` | `scripts/features/embed.py --finetuned` | Vocabulary caches of the finetuned models | 132 GB |
| `fonts/` | `src/utils/viz.py` | The bold face cut from Noto Sans for the figures, written by the first figure | 0.4 MB |
| `notebooks/` | `scripts/run_analysis.sh` | The executed notebooks with their printed outputs | 2 MB |

## Recorded names

Some folder names are older than the paper's terms. `concepts/` holds the
class-name experiment and `composition/` the two-clue experiment. They keep
these names because the manifests record image paths that start with them,
the manifest fingerprints hash those paths, and every embedding cache and
result table is checked against a fingerprint. Renaming a folder would make
every cache built on it stale.

`lexicon/` here holds tables generated from the committed `lexicon/` in the
repository root. The committed folder holds the word lists as written; this
one holds the word, pair and audit tables built from them and the manifest of
their renders.

The vocabulary and two-clue manifests record absolute paths, such as
`/data/agjma/vit_semantic/renders/...`, and the fingerprint covers them. The
fingerprints are therefore specific to the machine that rendered the images.
A fresh run of the stimulus stage elsewhere gives different `manifest_sha`
values in the caches and result tables, even where every number is the same.
`src/config.py` splits off the recorded prefix when it opens an image, so the
recorded caches still load under a different data root.
