# Vision Transformers Learn Word Meaning from Images Alone

Anders Gjølbye\*, Sebastian Ray Mason\*, Stella Frank, Lars Kai Hansen<br>
Technical University of Denmark<br>
<sub>\* Equal contribution</sub>

Can a vision model learn what words mean from pixels? DINOv3 is trained
without captions, class labels or any other language supervision. Writing
appears in some of its training images, but only as pixels. We render words on
images, embed the images with fourteen vision transformers and find that the
two largest DINOv3 models place related words close together.

<p align="center">
  <img src="assets/nearest_words.png" width="100%" alt="Left: the three nearest words to germany, petrol and porsche in five vision models, with cosine similarities. DINOv3-H ranks berlin, gasoline and ferrari first. Right: six countries in DINOv3-H, each joined to its capital by an arrow, hand-picked from 60 pairs to illustrate the country-to-capital offset.">
</p>

<sub>**Left:** the three nearest words to *germany*, *petrol* and *porsche* in
five models. Numbers are cosine similarities between word directions, defined
[below](#how-it-works), and bold marks a capital, a synonym or a brand from the
same industry. **Right:** six of the 60 country-capital pairs in DINOv3-H,
hand-picked to illustrate the offset from country to capital. The panel shows
the relation and does not measure how consistent it is.</sub>

In DINOv3-H, the nearest word to *germany* is *berlin*, the nearest to *petrol*
is *gasoline* and the nearest to *porsche* is *ferrari*. Ten of the other
thirteen models, including DINOv3-7B, CLIP-B and an untrained ViT-B, rank
*patrol* first, a look-alike that differs from *petrol* by one letter.

The fourteen models differ in what they were trained on.

| Training | Models |
|---|---|
| Images only, self-supervised | DINOv3-B, -L, -H and -7B, Web-DINO-1B, DINOv2-B, -L and -G, MAE-B, -L and -H |
| Image-caption pairs | CLIP-B, SigLIP2-B |
| None, a baseline for visual form | Untrained ViT-B |

For comparison, three Qwen3 text encoders embed the same words as text, and the
w2v-BERT 2.0 speech encoder embeds them as synthesized speech.

## Findings

- In DINOv3-H and DINOv3-7B, 60% of countries have their capital among their
  ten nearest words, against 55% in CLIP-B and 0% in DINOv2-G. Nouns and brands
  lie near words of their own category about as often as in CLIP-B.
- Synonyms are found less reliably. A word's synonym is among its ten nearest
  words for 6% of words in DINOv3-H, 10% in DINOv3-7B and 20% in CLIP-B,
  against 69% to 74% in the text encoders.
- DINOv3-L shows a weaker version of the same structure. The eight other
  self-supervised models, DINOv3-B, Web-DINO-1B and the DINOv2 and MAE models,
  do no better on these measures than the untrained ViT-B.
- In DINOv3-H, the word direction of an ImageNet class name ranks that class
  first among all 1,000 classes for over half of the 200 class names.
- DINOv3-H and DINOv3-7B share 12% to 14% of their ten nearest words with the
  text encoders, more than any other self-supervised vision model.
- Contrastive finetuning on COCO images with written or spoken captions, after
  removing images with detected text, raises synonym retrieval and the grouping
  of nouns by category in DINOv3-H. It lowers DINOv3-H's capital retrieval and
  changes the other nine finetuned models little.

The numbers come from Tables 1 and 2 and Figure 3 of the paper, which the
notebooks in `notebooks/` reproduce.

## How it works

<p align="center">
  <img src="assets/word_and_class_directions.png" width="100%" alt="Top row: ImageNet images of tigers, averaged into a visual direction. Bottom row: the word tiger rendered on three background images, each beside a control image with the letters scrambled; the differences are averaged into a word direction. A cosine similarity compares the two directions.">
</p>

*f* is the model's embedding of an image, and *x<sub>i</sub>* is one of a
class's 50 validation images. *x<sub>s</sub> ⊕ w* is word *w* rendered on
background *x<sub>s</sub>*, and *x<sub>s</sub> ⊕ w̃<sub>s</sub>* is its
control. A hat marks an average before centering. Centering gives the visual
direction ν and the word direction τ.

Each word is rendered once on each of 96 background images, with the font,
size, color and position set per background. Its control image has the same
background, typography and ink color, but the letters' opacity mask is
replaced by a phase-scrambled copy, so the ink stays in the same box and
nothing is readable. The model embeds both images. The difference between the
two embeddings, averaged over the 96 backgrounds and centered by subtracting
the mean over the vocabulary, is the *word direction* of that word (bottom
row).

The vocabulary has 1,481 words and names and 80 pseudowords in eight lists.
Among them are synonyms with a look-alike competitor (*petrol*, *gasoline*,
*patrol*), country-capital pairs, 384 nouns in six categories and 192 brands in
six industries. Two words are compared by the cosine similarity of their word
directions. Partner retrieval asks whether a word's partner, such as its
synonym or capital, is among its ten nearest words, out of all 1,560 other
items.

A class's *visual direction* is the mean embedding of its ImageNet-1k
validation images, centered over the 1,000 classes (top row). A second
vocabulary of 200 ImageNet class names, rendered the same way and centered over
those 200 names, tests whether each name's word direction is closest to the
visual direction of the class it names.

## Getting started

The repository holds the code and the word lists. Datasets, caches and trained
weights are not included.

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
  for the figures (the other three rendering fonts, DejaVu Sans, DejaVu Serif
  and DejaVu Sans Mono, come with Matplotlib)
- a Hugging Face account with access to the gated DINOv3 models and
  ImageNet-1k, signed in through the Hugging Face CLI

Two datasets, about 166 GB together, go in `data/`: the ImageNet-1k validation
images and SPEECH-COCO, which holds the COCO images with written and spoken
captions.

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
    --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 \
    --include "data/validation-*" --include classes.py \
    --local-dir data/imagenet-1k

hf download mteb/SpeechCoco --repo-type dataset \
    --revision ba337542de8f4ee66cfc285b5eeb69c326dc1576 \
    --include "data/*" --local-dir data/spokencoco_hf
```

To keep the data somewhere else, set `VIT_SEMANTIC_DATA` and change the
`--local-dir` paths to match. Model weights download on first use.

## Reproducing the paper

The four stages run in order, each reading what the one before wrote.

| Stage | Command | Hardware | Time |
|---|---|---|---|
| Render the word images and controls | `bash scripts/run_stimuli.sh` | CPU | a few hours |
| Embed them with every model and encoder | `bash scripts/run_features.sh` | GPU | 1-2 days |
| Finetune ten models on SPEECH-COCO, once with written and once with spoken captions | `bash scripts/run_finetune.sh` | GPU | 2-3 days |
| Compute the results, figures and tables | `bash scripts/run_analysis.sh` | CPU | about an hour |

Each `run_*.sh` script lists its inputs at the top and says which steps resume
after an interruption. The analysis stage runs the notebooks and writes figures
to `data/figures/`, LaTeX tables to `data/tables/` and result tables to
`data/results/` and `data/composition/results/`.

| Notebook | Paper |
|---|---|
| `figure1_neighbours` | Figure 1 |
| `figure2_datasets` | Figure 2 |
| `figure3_language_alignment` | Figure 3 |
| `figure4_rendering` | Figure 4 |
| `figure5_composition` | Figure 5, Tables 11 and 12 |
| `figure6_finetune` | Figure 6, Table 15 |
| `table1_models` | Tables 1 and 4 to 7 |
| `table2_concepts` | Tables 2 and 8 to 10 |
| `table3_wordlists` | Table 3 |
| `table13_feature_models` | Tables 13 and 14 |

<details>
<summary>Notes for an exact reproduction</summary>

- The recorded run used espeak-ng 1.51, PT Sans 2014-11-21, URW base35
  2020-09-10 and Noto Sans 2024-04-01 (`NotoSans[wght].ttf`).
- `environment.yml` pins the direct Python dependencies.
  `pip install -e ".[dev]"` in a Python 3.13 environment installs the same
  packages without pins.
- Pretrained models load their default Hugging Face revision.
- Run the stimulus stage once per data directory. A second run can choose
  different backgrounds without the later stages detecting it.
- Training on other hardware or library versions can keep the checkpoint from
  a different step. Before embedding a retrained model, set its step in
  `_FINETUNES` in [src/models.py](src/models.py). A smaller GPU needs a smaller
  micro-batch, set in [src/finetune/recipe.py](src/finetune/recipe.py), which
  can also change the step kept.

</details>

## Repository layout

| Directory | Contents |
|---|---|
| `lexicon/` | The eight word lists, the 200 class names, and the two-clue items, pairs of words that together point to a third (*input* and *sound* for *microphone*) |
| `src/` | Rendering, model loading, measurements and statistics |
| `scripts/` | The four stages, one folder each, with a `run_*.sh` script per stage |
| `notebooks/` | One notebook per figure or table group of the paper |

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
