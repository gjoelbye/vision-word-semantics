"""Contrastive finetuning on SPEECH-COCO, the third stage, run by `scripts/finetune/`.

`data` reads the training corpus and its caption targets. `model` joins a
backbone to a frozen caption tower, the text or speech encoder, and holds the
InfoNCE objective. `training` holds the recipe and the loop, and `checkpoint`
the on-disk format that `src.models.load` reads back for the feature stage.
"""
