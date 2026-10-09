"""Stimulus construction, the first stage, run by `scripts/stimuli/`.

`vocabulary` reads the committed lexical vocabulary and checks it,
`typography` holds the fonts, the 24 typography settings and the 96 slots,
and `render` draws a word set and its scrambled controls and writes the
manifest. `imagenet` covers the class names and their photographs, and
`two_clue_items` and `two_clue_render` the two-clue items. A script in
`scripts/stimuli/` may share a module's name, as `render` and `vocabulary` do.
"""
