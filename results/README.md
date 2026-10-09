# Results

The result tables the notebooks read, written by the analysis scripts in
`scripts/analysis/` from the caches under `data/`. Each file records the
fingerprints of the inputs it was computed from: `manifest_sha` for the
vocabulary manifest, `word_manifest_sha` and `image_manifest_sha` for the
class-name and class-image manifests, `word_sha` for the ordered word set the
text and speech encoders embedded, and `dataset_sha` and `manifest_sha` for the
two-clue items and their manifest. `src/paper/results.py` refuses a file whose
fingerprints differ from those of the current inputs.

| File | Written by | Paper | One row per | Columns |
|---|---|---|---|---|
| `neighbours.parquet` | `figure1_neighbours.py` | Figure 1a | model, query word and neighbour rank | `model`, `pool`, `query`, `rank`, `neighbour`, `cosine`, `semantic_partner`, `same_category`, `manifest_sha` |
| `relation_offsets.parquet` | `figure1_neighbours.py` | Offset values printed by `figure1_neighbours` | model | `model`, `relation`, `n_pairs`, `first_without_offset`, `first_with_offset`, `alignment_mean`, `alignment_mean_null_mean`, `alignment_mean_null_p`, `n_folds`, `n_perm`, `manifest_sha` |
| `figure1b_projection.parquet` | `figure1_neighbours.py` | Figure 1b | plotted word of DINOv3-H, plus one diagnostic row | `record_type`, `model`, `selected_pairwise_cosine`, `all_pairwise_cosine`, `n_display`, `n_undisplayed`, `pair_index`, `source`, `target`, `endpoint`, `word`, `x`, `y`, `manifest_sha` |
| `language_alignment.parquet` | `figure3_language_alignment.py` | Figure 3 | reference model or encoder, model and measure | `reference`, `model`, `measure`, `value`, `chance`, `length_null_mean`, `p`, `p_bh`, `n_perm`, `word_sha`, `manifest_sha` |
| `two_clue_summary.parquet` | `figure5_two_clue.py` | Figure 5 | model and analysis set | `model`, `analysis_set`, `n_families`, `n_equations`, `joint_margin`, `interaction_margin`, `feature_axis_gain_over_base`, `base_axis_gain_over_feature`, `interaction_margin_ci_lo`, `interaction_margin_ci_hi`, `feature_axis_gain_over_base_ci_lo`, `feature_axis_gain_over_base_ci_hi`, `base_axis_gain_over_feature_ci_lo`, `base_axis_gain_over_feature_ci_hi`, `joint_margin_p_greater`, `feature_axis_gain_over_base_p_greater`, `base_axis_gain_over_feature_p_greater`, `interaction_margin_p_two_sided`, `structured_iut_effect`, `structured_iut_p_greater`, `joint_margin_p_holm`, `structured_iut_p_holm`, `interaction_margin_p_holm`, `n_bootstrap`, `bootstrap_seed`, `dataset_sha`, `manifest_sha` |
| `finetune_effect.parquet` | `figure6_finetune.py` | Figure 6 | model or finetune, readout, task and number of word images | `model`, `readout`, `task`, `k`, `images`, `draws`, `value`, `spread`, `manifest_sha`, `representation`, `estimator`, `layer_index` |
| `finetune_draws.parquet` | `figure6_finetune.py` | Draws printed by `figure6_finetune` | number of word images and draw | `images`, `draw`, `appearance`, `font`, `placement`, `size`, `color`, `box`, `manifest_sha` |
| `retrieval.parquet` | `table1_models.py` | Tables 1, 4, 5 and 7, Figure 3, Section 5 | model, relation and query direction | `model`, `kind`, `relation`, `direction`, `n`, `hits`, `hits10`, `rate10`, `rate10_se`, `chance10`, `manifest_sha` |
| `purity.parquet` | `table1_models.py` | Tables 1, 6 and 7, Section 5 | model and list | `model`, `axis`, `n`, `purity`, `se`, `p`, `chance`, `n_perm`, `manifest_sha` |
| `letter_cluster.parquet` | `table1_models.py` | Values printed by `table1_models` | model | `model`, `axis`, `within_cluster`, `within_cluster_chance`, `manifest_sha` |
| `items.parquet` | `table1_models.py` | Table 5 | model and comparison | `model`, `record_type`, `block`, `n`, `wins`, `sign_p`, `kept_above_baseline`, `kept_above_baseline_se`, `other_above_baseline`, `other_above_baseline_se`, `concrete_found`, `abstract_found`, `n_concrete`, `n_abstract`, `fisher_p`, `sign_p_holm`, `manifest_sha` |
| `latent_categories.parquet` | `table1_models.py` | Table 6 | model and list | `model`, `dataset`, `n_test`, `n_correct`, `full_classification_accuracy`, `full_classification_accuracy_se`, `classification_chance`, `length_stratified_full_classification_accuracy_p`, `n_perm`, `manifest_sha` |
| `depth.parquet` | `table1_models.py` | Table 7 | model and depth | `model`, `layer`, `layer_index`, `n_layers`, `relative_depth`, `purity`, `purity_se`, `retrieval`, `retrieval10`, `retrieval10_se`, `manifest_sha` |
| `encoder_lexical.parquet` | `table1_models.py` | Encoder rows of Tables 1, 4, 5 and 6 | encoder, readout, word group and query direction | `model`, `estimator`, `readout`, `group`, `direction`, `estimate`, `se`, `n`, `m`, `hits10`, `chance`, `method`, `word_sha`, `manifest_sha` |
| `centering.parquet` | `table1_models.py` | Centering values printed by `table3_wordlists` | model | `model`, `shared_mean_share`, `manifest_sha` |
| `class_alignment.parquet` | `table2_class_names.py` | Tables 2 and 8 | model and image subset | `model`, `subset`, `n_classes`, `n_words`, `alignment`, `se`, `null_mean`, `p`, `n_permutations`, `word_manifest_sha`, `image_manifest_sha` |
| `class_retrieval.parquet` | `table2_class_names.py` | Tables 2 and 9 | model, image subset and retrieval direction | `model`, `subset`, `direction`, `n`, `n_candidates`, `hits`, `hits10`, `rate10`, `rate10_se`, `chance10`, `median_rank`, `word_manifest_sha`, `image_manifest_sha` |
| `class_checks.parquet` | `table2_class_names.py` | Background check printed by `table2_class_names` | model | `model`, `n_words`, `max_backgrounds_per_word`, `with_all_slots`, `without_own_slot`, `word_manifest_sha`, `image_manifest_sha` |
| `class_diagnostics.parquet` | `table2_class_names.py` | Class diagnostics printed by `table2_class_names` | model and ImageNet class | `model`, `wnid`, `word`, `coherence`, `nearest_wnid`, `nearest_word`, `nearest_cosine`, `word_manifest_sha`, `image_manifest_sha` |
| `class_depth.parquet` | `table2_class_names.py` | Table 10 | model and depth | `model`, `layer`, `layer_index`, `n_layers`, `relative_depth`, `alignment`, `word_to_class_rate10`, `word_to_class_chance10`, `word_manifest_sha`, `image_manifest_sha` |
