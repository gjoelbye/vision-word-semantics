# Results

The result tables the notebooks read, written by the analysis scripts in
`scripts/analysis/` from the caches under `data/`. Each file records the
fingerprints of the manifests it was computed from: `manifest_sha` for the
vocabulary, `word_manifest_sha` and `image_manifest_sha` for the class names
and class images, and `dataset_sha` and `manifest_sha` for the two-clue items.
`src/utils/results.py` refuses a file whose fingerprints differ from those of
the current manifests.

| File | Written by | Paper | One row per |
|---|---|---|---|
| `neighbours.parquet` | `figure1_neighbours.py` | Figure 1a | model, query word and neighbour rank |
| `relation_offsets.parquet` | `figure1_neighbours.py` | Offset values printed by `figure1_neighbours` | model |
| `figure1b_projection.parquet` | `figure1_neighbours.py` | Figure 1b | plotted word of DINOv3-H, plus one diagnostic row |
| `language_alignment.parquet` | `figure3_language_alignment.py` | Figure 3 | reference model or encoder, model and measure |
| `two_clue_summary.parquet` | `figure5_composition.py` | Figure 5 | model and analysis set |
| `finetune_effect.parquet` | `figure6_finetune.py` | Figure 6 | model or finetune, readout, task and number of word images |
| `finetune_draws.parquet` | `figure6_finetune.py` | Draws printed by `figure6_finetune` | number of word images and draw |
| `retrieval.parquet` | `table1_models.py` | Tables 1, 4, 5 and 7, Figure 3 | model, relation and query direction |
| `purity.parquet` | `table1_models.py` | Tables 1, 6 and 7 | model and list |
| `letter_cluster.parquet` | `table1_models.py` | Values printed by `table1_models` | model |
| `items.parquet` | `table1_models.py` | Table 5 | model and comparison |
| `latent_categories.parquet` | `table1_models.py` | Table 6 | model and list |
| `depth.parquet` | `table1_models.py` | Table 7 | model and depth |
| `encoder_lexical.parquet` | `table1_models.py` | Encoder rows of Tables 1, 4, 5 and 6 | encoder, readout, word group and query direction |
| `centering.parquet` | `table1_models.py` | Centering values printed by `table3_wordlists` | model |
| `concept_alignment.parquet` | `table2_concepts.py` | Tables 2 and 8 | model and image subset |
| `concept_retrieval.parquet` | `table2_concepts.py` | Tables 2 and 9 | model, image subset and retrieval direction |
| `concept_checks.parquet` | `table2_concepts.py` | Background check printed by `table2_concepts` | model |
| `concept_classes.parquet` | `table2_concepts.py` | Class diagnostics printed by `table2_concepts` | model and ImageNet class |
| `concept_depth.parquet` | `table2_concepts.py` | Table 10 | model and depth |
