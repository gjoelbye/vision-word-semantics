"""The lexical vocabulary read as text by the frozen Qwen3 embedding models.

Each word is embedded inside the same 96 written carriers and the vectors are
averaged. Rows follow the vocabulary order of the word directions.
`scripts/features/word_text.py` writes one cache per encoder, and the
similarity script and the analysis read it through `load`. The finetuning code
reuses `Qwen3Tower` and `tokenizer` for its caption targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import WORD_TEXT, line_fingerprint
from ..models import ENCODERS

# Token length every carrier is padded to. `embed` checks on every batch
# that a pad token survives, so no carrier is truncated.
CONTEXT = 48

# The text encoders, by size.
TEXT_MODELS = ("qwen3-0p6b-frozen", "qwen3-4b-frozen", "qwen3-8b-frozen")

# The 96 shared carriers, as 24 frames crossed with 4 surface variants.
# Every target slot is a quoted, named or otherwise permissive position, so
# each carrier is grammatical for every word in the vocabulary.
TEMPLATE_FRAMES = (
    ("bare", ("{w}", "{w}.", "{w}!", "{w}?")),
    ("bare", ('"{w}"', "'{w}'", "({w})", "[{w}]")),
    ("metalinguistic", ('the word "{w}"', 'the term "{w}"',
                        'a word: "{w}"', 'the string "{w}"')),
    ("metalinguistic", ("The word is {w}.", "The word was {w}.",
                        "This word is {w}.", "The word is {w}!")),
    ("metalinguistic", ("Read {w} aloud.", "Read {w} out loud.",
                        "Please read {w} aloud.", "Read {w} aloud!")),
    ("metalinguistic", ("Spell {w}.", "Spell out {w}.",
                        "Please spell {w}.", "Spell {w}!")),
    ("metalinguistic", ("{w} is spelled out.", "{w} was spelled out.",
                        "{w} is spelled aloud.", "{w} is spelled out!")),
    ("predicative", ("It says {w} there.", "It said {w} there.",
                     "It says {w} here.", "It says {w} there!")),
    ("predicative", ("{w} is what it says.", "{w} was what it said.",
                     "{w} is what it reads.", "{w} is what it says!")),
    ("predicative", ("This one is {w}.", "That one is {w}.",
                     "This one was {w}.", "This one is {w}!")),
    ("predicative", ("That would be {w}.", "That will be {w}.",
                     "This would be {w}.", "That would be {w}!")),
    ("predicative", ("The answer is {w}.", "The answer was {w}.",
                     "An answer is {w}.", "The answer is {w}!")),
    ("reported speech", ("They wrote {w}.", "He wrote {w}.",
                         "She wrote {w}.", "They write {w}.")),
    ("reported speech", ("Someone said {w}.", "Someone says {w}.",
                         "Somebody said {w}.", "Someone said {w}!")),
    ("reported speech", ("She read {w} out.", "He read {w} out.",
                         "They read {w} out.", "She reads {w} out.")),
    ("reported speech", ("He typed {w} quickly.", "She typed {w} quickly.",
                         "He typed {w} slowly.", "He types {w} quickly.")),
    ("reported speech", ("I heard {w} again.", "We heard {w} again.",
                         "I hear {w} again.", "I heard {w} once more.")),
    ("listing and discourse", ("{w}, and then {w}.", "{w}, then {w}.",
                               "{w} and then {w}.", "{w}, and after that {w}.")),
    ("listing and discourse", ("First {w}. Then the rest.",
                               "First {w}. Then the others.",
                               "First {w}, then the rest.",
                               "First came {w}. Then the rest.")),
    ("listing and discourse", ("{w}, {w}, {w}.", "{w}, {w}.",
                               "{w}; {w}; {w}.", "{w}, {w}, {w}, {w}.")),
    ("listing and discourse", ("Next up: {w}.", "Next: {w}.",
                               "Up next: {w}.", "Next up, {w}.")),
    ("listing and discourse", ("Here it is, {w}.", "There it is, {w}.",
                               "Here it was, {w}.", "Here it is: {w}.")),
    ("caption", ("A sign showing {w}.", "A sign reading {w}.",
                 "The sign showing {w}.", "A sign showed {w}.")),
    ("caption", ("Written on it: {w}.", "Written on it, {w}.",
                 "Printed on it: {w}.", "Written there: {w}.")),
)

# Variants per frame.
VARIANTS = 4

CARRIERS = tuple(carrier for _, group in TEMPLATE_FRAMES for carrier in group)

# Frame index of each carrier.
CARRIER_FRAME = np.array([i for i, _ in enumerate(TEMPLATE_FRAMES)
                          for _ in range(VARIANTS)])

assert len(CARRIERS) == len(set(CARRIERS)) == 96, "carriers must be 96 distinct"


# sha256 of the word list in order, recorded with every text and speech cache.
fingerprint = line_fingerprint


def cache_dir(key: str) -> Path:
    return WORD_TEXT / key


class Qwen3Tower(nn.Module):
    """A frozen Qwen3 embedding model, pooled at the last token.

    The tokenizer pads on the left, so the last position of every row is the
    last real token.
    """

    def __init__(self, repo: str) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            repo, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        # Frozen means eval mode always, so dropout never perturbs the output.
        super().train(False)
        self.model.eval()
        return self

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.model(**batch).last_hidden_state
        return out[:, -1, :].float()


def tokenizer(key: str, context: int):
    """Text to left-padded token tensors of length `context` for a Qwen3 encoder."""
    repo, _, kind = ENCODERS[key]
    if kind != "qwen3":
        raise KeyError(f"encoder kind {kind!r} does not tokenise text")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return lambda texts: dict(tok(
        list(texts), padding="max_length", truncation=True,
        max_length=context, return_tensors="pt"))


def embed(key: str, words, *, device: str = "auto", batch_size: int = 64
          ) -> tuple[np.ndarray, str]:
    """`(per-carrier vectors, device used)`, shaped `(words, 96, width)`.

    The mean over axis 1 is the word's vector. `device` of `auto` uses the
    GPU and falls back to the CPU when the weights do not fit.
    """
    words = list(words)
    tokenize = tokenizer(key, CONTEXT)
    # Every carrier is applied to every word, so the averaging weights do not
    # depend on the word.
    texts = [carrier.format(w=word) for word in words for carrier in CARRIERS]
    assert len(texts) == len(words) * len(CARRIERS), "the design is not crossed"

    def run(where: str) -> np.ndarray:
        tower = Qwen3Tower(ENCODERS[key][0])
        tower = tower.float() if where == "cpu" else tower
        tower = tower.to(where).eval()
        out = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = tokenize(texts[start:start + batch_size])
                if not (batch["attention_mask"][:, 0] == 0).all():
                    raise ValueError(
                        f"a carrier filled the {CONTEXT}-token context and may "
                        f"have been truncated, which would make the averaging "
                        f"weights word-dependent; raise text.CONTEXT")
                batch = {k: v.to(where) for k, v in batch.items()}
                with torch.no_grad():
                    out.append(tower(batch).float().cpu().numpy())
        finally:
            del tower
            if where != "cpu":
                torch.cuda.empty_cache()
        stacked = np.concatenate(out).astype(np.float32)
        return stacked.reshape(len(words), len(CARRIERS), -1)

    if device != "auto":
        return run(device), device
    if not torch.cuda.is_available():
        return run("cpu"), "cpu"
    try:
        return run("cuda"), "cuda"
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{key} does not fit in the free GPU memory; using the processor")
        return run("cpu"), "cpu"


def crossing_gap(per_carrier: np.ndarray) -> tuple[float, float]:
    """`(within-frame, between-frame)` mean similarity of carrier signatures.

    A carrier's signature is the mean of its word vectors, centered over
    carriers. Variants of one frame are expected to be more alike than variants
    of different frames.
    """
    signature = per_carrier.mean(0)
    signature = signature - signature.mean(0, keepdims=True)
    signature /= np.linalg.norm(signature, axis=1, keepdims=True) + 1e-9
    similarity = signature @ signature.T
    same = CARRIER_FRAME[:, None] == CARRIER_FRAME[None, :]
    other = ~np.eye(len(similarity), dtype=bool)
    return float(similarity[same & other].mean()), float(similarity[~same].mean())


def load(key: str, words) -> np.ndarray:
    """The cached mean over carriers.

    Refused when the recorded word fingerprint or carrier list differs from the
    current ones.
    """
    directory = cache_dir(key)
    meta_file = directory / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"no text cache for {key}; run scripts/features/word_text.py")
    meta = json.loads(meta_file.read_text())
    want = fingerprint(words)
    if meta.get("word_sha") != want:
        raise RuntimeError(
            f"{key} was embedded for a different word set "
            f"({str(meta.get('word_sha'))[:12]} != {want[:12]}); "
            f"re-run scripts/features/word_text.py")
    if list(meta.get("carriers", [])) != list(CARRIERS):
        raise RuntimeError(
            f"{key} was embedded with a different carrier set; "
            f"re-run scripts/features/word_text.py")
    vectors = np.load(directory / "vectors.npy")
    if len(vectors) != len(words):
        raise RuntimeError(f"{key} holds {len(vectors)} rows for {len(words)} words")
    return vectors
