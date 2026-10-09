# Vision Transformers Learn Word Meaning from Images Alone

**Technical University of Denmark**

Anders Gjølbye\*, Sebastian Ray Mason\*, Stella Frank, Lars Kai Hansen (\*equal contribution)

DINOv3 is trained on images alone, without captions, class labels or any other
language supervision. We render words on images, embed the images with
fourteen vision transformers and find that the two largest DINOv3 models still
place related words close together. This repository holds the code and word
lists behind the paper.

<p align="center">
  <img src="assets/nearest_words.png" width="100%" alt="Nearest words to germany, petrol and porsche in five models, and six country-capital pairs in DINOv3-H.">
</p>

Left: the nearest words in five models, with cosine similarities. Bold marks a
capital, a synonym or a brand from the same industry. Right: six hand-picked
country-capital pairs in DINOv3-H.

## Main results

- In DINOv3-H and DINOv3-7B, 60% of countries have their capital among their
  ten nearest words, against 55% in CLIP-B, which learned from image-caption
  pairs. Nouns and brands group by category about as often as in CLIP-B.
- Synonyms are harder. A word's synonym is among its ten nearest words for 6%
  of words in DINOv3-H and 10% in DINOv3-7B, against 20% in CLIP-B and 69% to
  74% in text encoders. For *petrol*, DINOv3-H ranks *gasoline* first, while
  ten of the other thirteen models rank the look-alike *patrol* first.
- DINOv3-L shows a weaker version of the same structure. The eight other
  self-supervised models, DINOv3-B, Web-DINO-1B and the DINOv2 and MAE models,
  do no better on these measures than an untrained ViT-B.
- In DINOv3-H, a rendered ImageNet class name ranks its own class first among
  all 1,000 classes for over half of the 200 names tested.
- Finetuning on COCO images with written or spoken captions raises synonym
  retrieval and category grouping in DINOv3-H, lowers its capital retrieval and
  changes the other finetuned models little.

The numbers are from the paper.
[From paper to code](#from-paper-to-code) lists the notebook, script and
files behind each figure and table.

## How it works

<p align="center">
  <img src="assets/word_and_class_directions.png" width="100%" alt="Images of tigers averaged into a visual direction, and the word tiger on three backgrounds minus scrambled controls averaged into a word direction.">
</p>

Top: the visual direction of the class *tiger*. Bottom: the word direction of
its name, from word images and their scrambled controls.

Each word is rendered on 96 background images. Every word image has a control
image with the same background, typography and ink color, in which the letters
are phase-scrambled so that nothing is readable. The difference between the two
embeddings, averaged over the backgrounds and centered over the vocabulary, is
the word direction. Two words are compared by the cosine similarity of their
word directions, and partner retrieval asks whether a word's synonym or capital
is among its ten nearest words, out of 1,560. A class's visual direction is the
centered mean embedding of its ImageNet-1k validation images, and comparing it
with the word direction of the class name tests whether a name points to the
images it names.

Each measure is computed in one function, under `src/analysis/` unless a path is given:

| Measure | Code |
|---|---|
| Word direction | `directions.word_directions` |
| Visual direction | `directions.class_mean` |
| Cosine similarity between word directions | `lexical.similarity` |
| Partner retrieval | `lexical.retrieval`, from the ranks in `lexical.partner_rank` |
| Neighbor purity and its permutation test | `lexical.knn_purity`, `lexical.purity_permutation` |
| Comparisons within an item | `lexical.build_items`, scored by `items_rows` in `scripts/analysis/table1_models.py` |
| Nearest-centroid accuracy and its length-stratified permutation test | `lexical.nearest_centroid_accuracy`, tested by `latent_rows` in `scripts/analysis/table1_models.py` |
| Mutual 10-NN alignment and linear CKA | `measure_cells` in `scripts/analysis/figure3_language_alignment.py` |
| Split-half cosine | `directions.split_half_cosine` |
| Matched cosine similarity | `word_class.class_alignment` |
| Word-to-class and class-to-word retrieval | `word_class.class_retrieval` |
| Two-clue margins | `two_clue.condition_vectors`, `two_clue.target_metrics` |

## Models

| Training | Models |
|---|---|
| Images only, self-supervised | DINOv3-B, -L, -H and -7B, Web-DINO-1B, DINOv2-B, -L and -G, MAE-B, -L and -H |
| Image-caption pairs | CLIP-B, SigLIP2-B |
| None, a baseline for visual form | Untrained ViT-B |

For comparison, three Qwen3 text encoders embed the same words as text, and the
w2v-BERT 2.0 speech encoder embeds them as synthesized speech.

## Installation

```bash
git clone https://github.com/gjoelbye/vision-word-semantics.git
cd vision-word-semantics
conda env create -f environment.yml
conda activate vit-semantic
python -m nltk.downloader wordnet omw-1.4
```

A full run also needs:

- a GPU with 32 GB of memory and about 750 GB of free disk
- [espeak-ng](https://github.com/espeak-ng/espeak-ng), which turns the words
  into phonemes for the speech synthesizer
- the fonts PT Sans, Nimbus Roman and URW Gothic for rendering, and Noto Sans
  for the figures
- a Hugging Face account with access to the gated DINOv3 models and
  ImageNet-1k, signed in through the Hugging Face CLI

The repository holds no datasets, caches or trained weights. Model weights
download on first use.

## Data

The ImageNet-1k validation images and SPEECH-COCO, which holds the COCO images
with written and spoken captions, take about 166 GB together.
[data/README.md](data/README.md) gives the download commands and lists what
each stage writes under `data/`. To keep the data somewhere else, set
`VIT_SEMANTIC_DATA`.

## Reproducing the paper

The four stages run in order, each reading what the one before wrote.

| Stage | Command | Hardware | Time |
|---|---|---|---|
| Render the word images and controls | `bash scripts/run_stimuli.sh` | CPU | a few hours |
| Embed them with every model and encoder | `bash scripts/run_features.sh` | GPU | 1-2 days |
| Finetune ten models on SPEECH-COCO, with written and with spoken captions | `bash scripts/run_finetune.sh` | GPU | 2-3 days |
| Compute the results, figures and tables | `bash scripts/run_analysis.sh` | CPU | about an hour |

Each `run_*.sh` script lists its inputs at the top and says which steps resume
after an interruption. The analysis stage writes result tables to `results/`,
then runs the notebooks, which write figures to `figures/` and LaTeX tables to
`tables/`. All three folders are committed. The notebooks also read the
manifests and word tables under `data/`, so they run only after the stages
before them.

### From paper to code

Each figure and table comes from one notebook in `notebooks/`. Where an
analysis script in `scripts/analysis/` computes its numbers first, the script
writes the listed result tables to `results/`. Tables 3 and 11 to 15 are typed
into the paper from what their notebook prints.

| Paper | Notebook | Analysis script | Result tables | Output |
|---|---|---|---|---|
| Figure 1a | `figure1_neighbours` | `figure1_neighbours.py` | `neighbours` | `tables/figure1a_neighbours.tex` |
| Figure 1b | `figure1_neighbours` | `figure1_neighbours.py` | `figure1b_projection` | `figures/figure1b_pairs.pdf` |
| Figure 2 | `figure2_datasets` | - | - | `figures/figure2_datasets.pdf` |
| Figure 3 | `figure3_language_alignment` | `figure3_language_alignment.py` | `language_alignment`, `retrieval` | `figures/figure3_language_alignment.pdf` |
| Figure 4 | `figure4_rendering` | - | - | `figures/figure4_rendering.pdf` |
| Figure 5 | `figure5_two_clue` | `figure5_two_clue.py` | `two_clue_summary` | `figures/figure5_two_clue.pdf` |
| Figure 6 | `figure6_finetune` | `figure6_finetune.py` | `finetune_effect`, `finetune_draws` | `figures/figure6_finetune.pdf` |
| Table 1 | `table1_models` | `table1_models.py` | `purity`, `retrieval`, `encoder_lexical` | `tables/table1_models.tex` |
| Table 2 | `table2_class_names` | `table2_class_names.py` | `class_alignment`, `class_retrieval` | `tables/table2_class_names.tex` |
| Table 3 | `table3_wordlists` | - | - | printed |
| Table 4 | `table1_models` | `table1_models.py` | `retrieval`, `encoder_lexical` | `tables/table4_lexical_retrieval.tex` |
| Table 5 | `table1_models` | `table1_models.py` | `items`, `retrieval`, `encoder_lexical` | `tables/table5_items.tex` |
| Table 6 | `table1_models` | `table1_models.py` | `purity`, `latent_categories`, `encoder_lexical` | `tables/table6_category.tex` |
| Table 7 | `table1_models` | `table1_models.py` | `depth`, `purity`, `retrieval` | `tables/table7_lexical_depth.tex` |
| Table 8 | `table2_class_names` | `table2_class_names.py` | `class_alignment` | `tables/table8_class_alignment.tex` |
| Table 9 | `table2_class_names` | `table2_class_names.py` | `class_retrieval` | `tables/table9_class_retrieval.tex` |
| Table 10 | `table2_class_names` | `table2_class_names.py` | `class_depth` | `tables/table10_class_depth.tex` |
| Tables 11 and 12 | `figure5_two_clue` | - | - | printed |
| Tables 13 and 14 | `table13_feature_models` | - | - | printed |
| Table 15 | `figure6_finetune` | - | - | printed, from the checkpoint records |

[results/README.md](results/README.md) lists the columns of every result
table and the other values each notebook prints.

### Notes on exact reproduction

- The recorded run used espeak-ng 1.51, PT Sans 2014-11-21, URW base35
  2020-09-10 and Noto Sans 2024-04-01 (`NotoSans[wght].ttf`).
- `environment.yml` pins the direct Python dependencies.
  `pip install -e ".[dev]"` in a Python 3.13 environment installs the same
  packages without pins.
- Pretrained models load their default Hugging Face revision.
- Run the stimulus stage once per data directory. A second run can choose
  different backgrounds, and later stages do not detect the change.
- Training on other hardware or library versions can keep the checkpoint from
  a different step. Before embedding a retrained model, set its step in
  `_FINETUNES` in [src/models.py](src/models.py). A smaller GPU needs a smaller
  micro-batch, set in [src/finetune/training.py](src/finetune/training.py), which
  can also change the step kept.

## Repository layout

| Directory | Contents |
|---|---|
| `lexicon/` | The word lists the paper uses: `vocabulary.tsv` with the eight lexical lists (1,481 words and names plus 80 pseudowords), `class_names.tsv` with the 384 candidate ImageNet-1k class names and the 200 kept, and the two-clue items, pairs of words that together point to a third; see [lexicon/README.md](lexicon/README.md) |
| `src/` | Library code with one folder per stage, `stimuli/`, `features/`, `finetune/` and `analysis/`, and `paper/` for the figures and tables; the paths, the model registry, the statistics and the text detector sit at the top |
| `scripts/` | The four stages, one folder each with the same names, and a `run_*.sh` script per stage |
| `notebooks/` | One notebook per figure or table group of the paper, committed without outputs |
| `results/` | The result tables the notebooks read; see [results/README.md](results/README.md) |
| `figures/` | The six figures as PDF |
| `tables/` | The generated LaTeX of Figure 1a and Tables 1, 2 and 4 to 10 |
| `data/` | Downloads and caches, not tracked; see [data/README.md](data/README.md) |
| `tests/` | Checks of the manifest row order, the paired difference, the split halves, centering and the readouts; run with `python -m pytest tests/ -q` |
| `assets/` | The images in this README |

## Citation

```bibtex
@misc{gjolbye2026vision,
  title  = {Vision Transformers Learn Word Meaning from Images Alone},
  author = {Gj{\o}lbye, Anders and Mason, Sebastian Ray and Frank, Stella and Hansen, Lars Kai},
  year   = {2026}
}
```

## License

The code and word lists are released under the [MIT License](LICENSE). The
class-name and two-clue files quote glosses from
[WordNet 3.0](https://wordnet.princeton.edu/license-and-commercial-use),
copyright Princeton University, under its license. The datasets and models
keep their own terms.
