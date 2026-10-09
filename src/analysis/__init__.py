"""Analysis code, the last stage, called by `scripts/analysis/`.

`directions` reads the class-token caches and builds the word and visual
directions. `lexical` holds the readouts of Section 4 and Appendix C,
`word_class` those of Section 5, and `two_clue` the two-clue scores of
Figure 5. Statistics shared across them are in `src.stats`.
"""
