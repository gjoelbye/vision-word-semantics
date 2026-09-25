# Vision Transformers Learn Word Meaning from Images Alone

Code for the paper. We test whether vision transformers group words rendered
on photographs by meaning, spelling or visual form, and how this changes
after finetuning on written or spoken captions.

The repository includes the curated word lists, experiment code and notebooks
for the paper's figures and tables. Datasets, feature caches and trained
checkpoints are not included. Notebook outputs are cleared for release.

## Setup

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate vit-semantic
python -m nltk.downloader wordnet omw-1.4
```

The environment pins the packages used for the recorded run and installs the
source in editable mode. Alternatively, use `pip install -e ".[dev]"` in a
Python 3.13 environment; this does not pin dependency versions. Keep the source
checkout: the package reads the word lists from `lexicon/`.

You also need:

- `espeak-ng` for speech synthesis (version 1.51 in the recorded run).
- PT Sans, Nimbus Roman and URW Gothic for rendering. DejaVu fonts come with
  Matplotlib. The recorded run used PT Sans 2014-11-21 and URW base 35 2020-09-10.
- Noto Sans variable font (`NotoSans[wght].ttf`, release 2024-04-01) for figures.
- A CUDA GPU with 32 GB memory and about 670 GB of disk space for a full run,
  including model downloads.

Models and EasyOCR weights download on first use. Request access to the four
DINOv3 repositories listed in [src/models.py](src/models.py), accept the
ImageNet-1k dataset terms on Hugging Face, then log in:

```bash
hf auth login
```

## Data

Download ImageNet-1k validation images and SPEECH-COCO (about 166 GB combined):

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
    --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 \
    --include "data/validation-*" --include classes.py \
    --local-dir data/imagenet-1k

hf download mteb/SpeechCoco --repo-type dataset \
    --revision ba337542de8f4ee66cfc285b5eeb69c326dc1576 \
    --include "data/*" --local-dir data/spokencoco_hf
```

SPEECH-COCO includes the COCO images, caption text and speech recordings;
no separate COCO download is needed.

Downloads and generated files live in `data/`, which is ignored by Git. Set
`VIT_SEMANTIC_DATA` to use another directory, and adjust the download paths
above to match.

## Reproduce the results

Run the four stages in order:

```bash
bash scripts/run_stimuli.sh     # CPU: a few hours
bash scripts/run_features.sh    # GPU: 1–2 days
bash scripts/run_finetune.sh    # GPU: 2–3 days
bash scripts/run_analysis.sh    # CPU: about an hour, including notebooks
```

The stages render the stimuli, extract features, train the twenty finetuned
models, and compute the paper's results. Each driver documents its resume
behavior; individual Python scripts accept `--help`. Set `PYTHON` to select
another interpreter.

Run stimulus generation once per data root. Rerunning background selection
can invalidate existing renders without changing their manifest fingerprints.
Feature caches and result tables are otherwise checked against their input
manifests.

Training can select a different checkpoint on different hardware or library
versions. Before embedding a retrained model, update its checkpoint step in
`_FINETUNES` in [src/models.py](src/models.py). Smaller GPUs need smaller
micro-batches in [src/finetune/recipe.py](src/finetune/recipe.py); this can also
change checkpoint selection. Pretrained models use the default Hugging Face
revision, rather than a pinned revision.

With the caches available, rerun only `scripts/run_analysis.sh`. It executes
the notebooks and writes figures to `data/figures/`, LaTeX tables to
`data/tables/`, and result tables to `data/results/`. Two-clue results are in
`data/composition/results/`.

## Find your way around

| Directory | Contents |
|---|---|
| `lexicon/` | Eight word lists, ImageNet class-name judgments and two-clue items |
| `src/` | Rendering, model loading, measurements and statistics |
| `src/finetune/` | Training code and settings |
| `scripts/` | Stimulus, feature, finetuning and analysis stages |
| `notebooks/` | Paper figures and tables |
| `tests/` | Checks of measurements, rendering alignment and statistics |

Notebook names follow the paper's figure and table numbers. `table1_models`
also produces Tables 4–7; `table2_concepts` produces Tables 8–10.
`figure5_composition` covers Tables 11–12, `table13_feature_models` covers
Tables 13–14, and `figure6_finetune` covers Table 15.

Paths and the global seed are set in [src/config.py](src/config.py); model
identifiers are in [src/models.py](src/models.py).

## Tests

```bash
python -m pytest tests/ -q
```

Two rendering tests require the backgrounds from the stimulus stage. The
remaining tests can run without downloaded datasets.

## License

Code and word lists: [MIT](LICENSE). Downloaded datasets and models retain
their own terms.
