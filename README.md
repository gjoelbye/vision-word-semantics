# Vision Transformers Learn Word Meaning from Images Alone

Code and word lists for the paper.

DINOv3 is trained with no language supervision at all. Yet when we render words
on images and embed them, the two largest DINOv3 models put related words close
together.

<p align="center">
  <img src="assets/nearest_words.png" width="100%" alt="Nearest words to germany, petrol and porsche in five models, and six country-capital pairs in DINOv3-H.">
</p>

Left: nearest words in five models, with cosine similarities. Bold marks a
related word. Right: six hand-picked country-capital pairs in DINOv3-H.

## Results

We compare fourteen vision transformers, and DINOv3-H and DINOv3-7B stand out.
In both, 60% of countries have their capital among their ten nearest words.
CLIP-B, trained on image-caption pairs, gets 55%. Nouns and brands group by
category about as often as in CLIP-B. Synonyms are harder: 6% in DINOv3-H and
10% in DINOv3-7B, against 20% in CLIP-B and 69% to 74% in text encoders. Most
models put *patrol* first for *petrol*. DINOv3-H puts *gasoline* first.

DINOv3-L shows a weaker version of this. The other eight self-supervised models
do no better on these measures than an untrained ViT-B.

Rendered class names also point to their classes. In DINOv3-H, the name of an
ImageNet class ranks that class first among all 1,000 for over half of the 200
names we tested. Finetuning on COCO with written or spoken captions raises
synonym retrieval and noun grouping in DINOv3-H, lowers its capital retrieval
and changes the other finetuned models little.

## How it works

<p align="center">
  <img src="assets/word_and_class_directions.png" width="100%" alt="Images of tigers averaged into a visual direction, and the word tiger on three backgrounds minus scrambled controls averaged into a word direction.">
</p>

Each word is rendered on 96 background images. Its control is the same image
with the letters phase-scrambled, so the ink is still there but nothing can be
read. A word's direction is the average difference between the two embeddings,
centered over the vocabulary. We compare words by the cosine similarity of
their directions and count a synonym or capital as found when it is among the
ten nearest of 1,560 words. For an ImageNet class we average the embeddings of
its validation images, center the mean over the 1,000 classes and compare the
result with the direction of the class name.

The code for these measures is in `src/analysis/`.

## Models

| Training | Models |
|---|---|
| Images only, self-supervised | DINOv3-B, -L, -H and -7B, Web-DINO-1B, DINOv2-B, -L and -G, MAE-B, -L and -H |
| Image-caption pairs | CLIP-B, SigLIP2-B |
| None (random weights) | Untrained ViT-B |

We also compare with three Qwen3 text encoders, given the words as text, and
with the w2v-BERT 2.0 speech encoder, given them as synthesized speech.

## Installation

```bash
git clone https://github.com/gjoelbye/vision-word-semantics.git
cd vision-word-semantics
conda env create -f environment.yml
conda activate vit-semantic
python -m nltk.downloader wordnet omw-1.4
```

`environment.yml` pins the versions we used. `pip install -e ".[dev]"` in
Python 3.13 installs the same packages without pins.

A full run also needs:

- a GPU with 32 GB of memory and about 750 GB of free disk
- [espeak-ng](https://github.com/espeak-ng/espeak-ng), for the speech synthesis
- the fonts PT Sans, Nimbus Roman, URW Gothic and Noto Sans
- access to the gated DINOv3 models and ImageNet-1k on Hugging Face, and a
  signed-in Hugging Face CLI

Model weights download on first use.

## Data

You need the ImageNet-1k validation images and SPEECH-COCO, about 166 GB
together. The download commands are in [data/README.md](data/README.md). To
keep the data somewhere else, set `VIT_SEMANTIC_DATA` to an absolute path.

## Reproducing the paper

Run the stages in this order:

```bash
bash scripts/run_stimuli.sh    # render word images and controls (CPU, a few hours)
bash scripts/run_features.sh   # embed them with every model (GPU, 1-2 days)
bash scripts/run_finetune.sh   # finetune ten models on SPEECH-COCO (GPU, 2-3 days)
bash scripts/run_analysis.sh   # results, figures and tables (CPU, about an hour)
```

The top of each script says what it needs and what it can resume.

The last stage writes result tables to `results/`, figures to `figures/` and
LaTeX tables to `tables/`. These folders are committed, so you can look at the
results without running anything. The notebooks in `notebooks/` draw the
figures and tables and print the numbers quoted in the text. Each is named
after the first figure or table it makes, and its first cell lists the others
and its inputs. [results/README.md](results/README.md) describes every result
table.

### Exact reproduction

- The recorded run used espeak-ng 1.51, PT Sans 2014-11-21, URW base35
  2020-09-10 and Noto Sans 2024-04-01.
- Hugging Face model revisions are not pinned.
- Run the stimulus stage only once per data directory. A rerun can pick
  different backgrounds, and nothing downstream will notice.
- The checkpoint kept for each finetune can change with hardware and library
  versions. If you retrain, put the new step into `_FINETUNES` in
  [src/models.py](src/models.py) before embedding. On a smaller GPU, lower
  `MICRO_BATCH` in [src/finetune/training.py](src/finetune/training.py), which
  can also change the step.

## Repository layout

```
lexicon/     word lists, class names and two-clue items
src/         library code, one folder per stage, and paper/ for figures and tables
scripts/     the four stages
notebooks/   one notebook per figure or table group
results/     result tables
figures/     paper figures
tables/      generated LaTeX tables
data/        downloads and caches, not tracked apart from its README
tests/
assets/      images for this README
```

[lexicon/](lexicon/README.md), [results/](results/README.md) and
[data/](data/README.md) have their own READMEs.

## Citation

```bibtex
@misc{gjolbye2026vision,
  title  = {Vision Transformers Learn Word Meaning from Images Alone},
  author = {Gj{\o}lbye, Anders and Mason, Sebastian Ray and Frank, Stella and Hansen, Lars Kai},
  year   = {2026},
  eprint = {2610.13252},
  archivePrefix = {arXiv}
}
```

## License

MIT, see [LICENSE](LICENSE). Glosses in the class-name and two-clue files come
from [WordNet 3.0](https://wordnet.princeton.edu/license-and-commercial-use)
and keep its license. Datasets and models keep their own terms.
