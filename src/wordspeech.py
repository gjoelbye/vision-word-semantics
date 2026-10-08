"""The word set spoken by a synthesizer and heard by a frozen speech encoder.

Each word is spoken in 96 renditions, eight voices by three speaking rates by
four carrier sentences, with the synthesizer's noise scales at zero. The
encoder's frames that cover the target word are pooled into four relative
bins, and the 96 vectors are averaged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import WORD_SPEECH
from .models import ENCODERS

#: The multi-speaker synthesizer. Its output is resampled to ``SAMPLE_RATE``.
TTS_REPO = "kakao-enterprise/vits-vctk"

SAMPLE_RATE = 16000

#: The speech encoder.
SPEECH_MODELS = ("w2v-bert-2.0-frozen",)

#: Relative-position bins the target's frames are pooled into, then
#: concatenated, so a cached vector is ``BINS`` times the encoder's width.
BINS = 4

#: The four spoken carriers. The target sits first, in the middle or last.
CARRIERS = (
    "{w} was written there.",
    "They wrote {w} on it.",
    "Someone said {w} out loud.",
    "The label reads {w}.",
)

#: Eight VCTK voices and three speaking rates.
SPEAKERS = (0, 13, 27, 40, 54, 67, 81, 94)
RATES = (0.9, 1.0, 1.1)

assert len(SPEAKERS) * len(RATES) * len(CARRIERS) == 96, "grid must be 96"


def grid() -> list[tuple[str, int, float]]:
    """The 96 renditions as ``(carrier, speaker, rate)``, in a fixed order."""
    return [(carrier, speaker, rate) for speaker in SPEAKERS for rate in RATES
            for carrier in CARRIERS]


def cache_dir(key: str) -> Path:
    return WORD_SPEECH / key


def token_span(tokenizer, carrier: str, word: str) -> tuple[int, int]:
    """The token indices covering ``word`` inside ``carrier.format(w=word)``.

    The tokenizer phonemizes and then interleaves a blank between every phoneme,
    so token ``2i + 1`` carries phoneme character ``i``. The target is located
    as the run of its own phonemes inside the carrier's phonemes.
    """
    def phonemes(text: str) -> str:
        ids = tokenizer(text).input_ids
        return "".join(t for t in tokenizer.convert_ids_to_tokens(ids)
                       if t != "_")

    prefix = carrier.split("{w}")[0]
    full = phonemes(carrier.format(w=word))
    target = phonemes(word)
    start = full.find(target, max(0, len(phonemes(prefix)) - 2) if prefix else 0)
    if start < 0:
        raise ValueError(f"{word!r} not found in the phonemisation of {carrier!r}")
    return 2 * start + 1, 2 * (start + len(target)) + 1


def synthesise(model, tokenizer, text: str, speaker: int, rate: float,
               span: tuple[int, int], device: str):
    """``(waveform at 16 kHz, sample range of the target)``.

    The per-token durations are read from the duration predictor through a
    forward hook, and they place the target inside the waveform.
    """
    import torch
    from scipy.signal import resample_poly

    captured = {}

    def hook(_module, _inputs, output):
        captured["log_duration"] = output

    handle = model.duration_predictor.register_forward_hook(hook)
    try:
        model.speaking_rate = rate
        batch = {k: v.to(device)
                 for k, v in tokenizer(text, return_tensors="pt").items()}
        with torch.no_grad():
            out = model(**batch, speaker_id=speaker)
    finally:
        handle.remove()

    mask = batch["attention_mask"].unsqueeze(1)
    duration = torch.ceil(torch.exp(captured["log_duration"]) * mask * (1.0 / rate))
    frames = duration[0, 0].detach().cpu().numpy().astype(np.int64)
    per_token = frames * int(np.prod(model.config.upsample_rates))
    edges = np.concatenate([[0], np.cumsum(per_token)])

    wave = out.waveform[0].float().cpu().numpy()
    # The recovered durations must sum to the length of the returned waveform.
    if int(edges[-1]) != len(wave):
        raise RuntimeError(
            f"recovered durations sum to {int(edges[-1])} samples but the "
            f"waveform is {len(wave)}; the duration hook is reading the wrong "
            f"quantity and every span would be misplaced")
    lo, hi = int(edges[span[0]]), int(edges[min(span[1], len(edges) - 1)])
    lo, hi = min(lo, len(wave)), min(hi, len(wave))

    native = model.config.sampling_rate
    if native != SAMPLE_RATE:
        ratio = SAMPLE_RATE / native
        wave = resample_poly(wave, SAMPLE_RATE // 50, native // 50)
        lo, hi = int(lo * ratio), int(hi * ratio)
        lo, hi = min(lo, len(wave)), min(hi, len(wave))
    return wave.astype(np.float32), (lo, max(hi, lo + 1))


def pool_span(hidden: np.ndarray, wave_length: int,
              span: tuple[int, int]) -> np.ndarray:
    """The target's frames, averaged within ``BINS`` consecutive bins and concatenated."""
    frames = len(hidden)
    lo = int(round(frames * span[0] / wave_length))
    hi = int(round(frames * span[1] / wave_length))
    lo = min(max(lo, 0), frames - 1)
    hi = min(max(hi, lo + 1), frames)
    piece = hidden[lo:hi]
    edges = np.linspace(0, len(piece), BINS + 1).astype(int)
    bins = [piece[max(a, 0):max(b, a + 1)].mean(0)
            for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate(bins).astype(np.float32)


def encode(features, tower, wave: np.ndarray, span: tuple[int, int],
           device: str) -> np.ndarray:
    """One vector for one utterance.

    Utterances are encoded one at a time, because padding inside a batch can
    change the number of encoder frames and so the bins.
    """
    import torch

    batch = features(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        hidden = tower(**batch).last_hidden_state[0].float().cpu().numpy()
    return pool_span(hidden, len(wave), span)


def alignment_score(tokenizer, tts, features, tower, probe, device) -> float:
    """Share of probe words whose two cut-outs identify each other.

    Each word is cut out of two different carriers, and a cut identifies its
    word when, after centering, it is nearest the other cut of the same word.
    """
    first, second = [], []
    for word in probe:
        for carrier, bucket in ((CARRIERS[0], first), (CARRIERS[1], second)):
            wave, span = synthesise(tts, tokenizer, carrier.format(w=word), 0,
                                    1.0, token_span(tokenizer, carrier, word),
                                    device)
            bucket.append(encode(features, tower, wave, span, device))

    def centred_unit(rows):
        m = np.stack(rows)
        m = m - m.mean(0, keepdims=True)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    similarity = centred_unit(first) @ centred_unit(second).T
    rank = (similarity > np.diag(similarity)[:, None]).sum(1)
    return float((rank == 0).mean())


def build(words, *, device: str = "cuda", verify: bool = True,
          probe: int = 24) -> np.ndarray:
    """Vectors of every rendition, shaped ``(words, 96, BINS * width)``.

    The mean over axis 1 is the word's vector.
    """
    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, VitsModel

    words = list(words)
    cells = grid()
    tokenizer = AutoTokenizer.from_pretrained(TTS_REPO)
    tts = VitsModel.from_pretrained(TTS_REPO).to(device).eval()
    tts.noise_scale = 0.0
    tts.noise_scale_duration = 0.0
    repo = ENCODERS[SPEECH_MODELS[0]][0]
    features = AutoFeatureExtractor.from_pretrained(repo)
    tower = AutoModel.from_pretrained(repo).to(device).eval()

    spans = {(carrier, word): token_span(tokenizer, carrier, word)
             for carrier in CARRIERS for word in words}
    print(f"located {len(spans)} target spans, no failures")

    if verify:
        # Probe words spread across the alphabetical vocabulary.
        chosen = [words[i] for i in np.linspace(0, len(words) - 1,
                                                min(probe, len(words))).astype(int)]
        share = alignment_score(tokenizer, tts, features, tower, chosen, device)
        print(f"alignment on {len(chosen)} words: a span cut from one carrier "
              f"identifies its own word against a span cut from another "
              f"{100 * share:.0f}% of the time")
        if share < 0.5:
            raise RuntimeError(
                "spans cut from two carriers mostly fail to identify each "
                "other, so the token-to-sample arithmetic is wrong")

    width = BINS * tower.config.hidden_size
    out = np.zeros((len(words), len(cells), width), np.float32)
    for index, word in enumerate(words):
        for cell, (carrier, speaker, rate) in enumerate(cells):
            wave, span = synthesise(tts, tokenizer, carrier.format(w=word),
                                    speaker, rate, spans[(carrier, word)], device)
            out[index, cell] = encode(features, tower, wave, span, device)
        if index % 50 == 0:
            print(f"  {index + 1}/{len(words)} words", flush=True)
    return out


def load(key: str, words) -> np.ndarray:
    """The cached mean over renditions.

    Refused when the recorded word fingerprint or carrier list differs from the
    current ones.
    """
    directory = cache_dir(key)
    meta_file = directory / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"no speech cache for {key}; run scripts/features/word_speech.py")
    meta = json.loads(meta_file.read_text())
    from .wordtext import fingerprint

    want = fingerprint(words)
    if meta.get("word_sha") != want:
        raise RuntimeError(
            f"{key} was embedded for a different word set "
            f"({str(meta.get('word_sha'))[:12]} != {want[:12]}); "
            f"re-run scripts/features/word_speech.py")
    if list(meta.get("carriers", [])) != list(CARRIERS):
        raise RuntimeError(
            f"{key} was embedded with a different carrier set; "
            f"re-run scripts/features/word_speech.py")
    vectors = np.load(directory / "vectors.npy")
    if len(vectors) != len(words):
        raise RuntimeError(f"{key} holds {len(vectors)} rows for {len(words)} words")
    return vectors
