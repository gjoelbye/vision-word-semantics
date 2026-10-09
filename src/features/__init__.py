"""Feature extraction, the second stage, run by `scripts/features/`.

`image` writes the class-token caches of every rendered image and photograph.
`text` and `speech` give each vocabulary word one vector from the frozen text
and speech encoders. Every cache records a fingerprint of its input, and the
readers refuse a cache whose fingerprint is not the current one.
"""
