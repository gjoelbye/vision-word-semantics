"""Feature extraction, the second stage, run by `scripts/features/`.

`image` writes the class-token caches of every rendered image and photograph.
`text` and `speech` give each vocabulary word one vector from the frozen text
and speech encoders. Every cache records a fingerprint of its input, and the
readers refuse a cache whose fingerprint is not the current one.
"""
from . import speech, text

# The estimator each kind of encoder is recorded under in the similarity index.
ESTIMATOR = {"text": text.ESTIMATOR, "speech": speech.ESTIMATOR}


def encoder_directions(key: str, words):
    """The vectors of one text or speech encoder for `words`, centered over words."""
    loader = speech if key in speech.SPEECH_MODELS else text
    vectors = loader.load(key, words)
    return vectors - vectors.mean(0, keepdims=True)
