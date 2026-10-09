"""Library code for the paper, grouped by pipeline stage.

The stage folders mirror `scripts/`: `stimuli` renders the words, `features`
embeds them, `finetune` trains the finetuned models, and `analysis` computes
the measures. `paper` turns the result tables into figures and LaTeX tables.
The modules at the top are shared: `config` holds the paths, `models` the model
registry and loader, `stats` the statistics, and `text_detector` the text scan
used by the class photographs and the finetuning corpus.
"""
