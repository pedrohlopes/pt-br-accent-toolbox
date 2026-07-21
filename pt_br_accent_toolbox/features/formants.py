from __future__ import annotations

from collections import defaultdict

import numpy as np
import parselmouth
import torch

from ..config import VOWEL_SET, VOWEL_MAP, SR, FRAME_MS
from ..alignment.zipa import (
    load_audio, load_vocab, make_fbank_extractor, make_session,
    utterance_logprobs, get_spikes,
)


FEAT_DIM = 3 * len(VOWEL_SET) + 6 + 2  # MF(21) + LTFD(6) + LTF0(2) = 29


def utterance_vowel_formants(
    audio: np.ndarray, sess, extractor, vocab: dict[int, str],
    max_formants: int = 5, max_freq: float = 5000.0,
    window_length: float = 0.025, pre_emphasis: float = 50.0,
) -> list[tuple[str, float, float, float, float]]:
    """
    Extract vowel formant tokens from one utterance.

    For each ZIPA vowel spike, query parselmouth Burg formants at that time.
    Returns list of (bp_vowel, F1, F2, F3, F0).
    """
    if len(audio) < 512:
        return []

    lp = utterance_logprobs(audio, sess, extractor)
    spikes = get_spikes(lp, vocab)

    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
    dur = snd.get_total_duration()
    fm = snd.to_formant_burg(
        max_number_of_formants=max_formants,
        maximum_formant=max_freq,
        window_length=window_length,
        pre_emphasis_from=pre_emphasis,
    )
    try:
        pit = snd.to_pitch()
    except Exception:
        pit = None

    out: list[tuple[str, float, float, float, float]] = []
    for t_fr, ph in spikes:
        bpv = VOWEL_MAP.get(ph)
        if bpv is None:
            continue
        ct = t_fr * FRAME_MS / 1000.0
        if ct <= 0 or ct >= dur:
            continue
        f1 = fm.get_value_at_time(1, ct)
        f2 = fm.get_value_at_time(2, ct)
        f3 = fm.get_value_at_time(3, ct)
        if not (np.isfinite(f1) and np.isfinite(f2) and np.isfinite(f3)):
            continue
        f0 = pit.get_value_at_time(ct) if pit is not None else np.nan
        out.append((bpv, float(f1), float(f2), float(f3), float(f0)))
    return out


def speaker_formant_vector(
    tokens: list[tuple[str, float, float, float, float]]
) -> tuple[np.ndarray, int]:
    """
    Aggregate per-vowel formant tokens into a 29D speaker vector.

    MF: per-BP-vowel mean F1/F2/F3 (21D)
    LTFD: mean/SD of F1/F2/F3 over all vowels (6D)
    LTF0: mean/SD of F0 (2D)
    Missing vowel -> NaN.

    Returns (feat[29], n_vowels).
    """
    mf = np.full(3 * len(VOWEL_SET), np.nan, np.float32)
    by = defaultdict(list)
    for v, f1, f2, f3, f0 in tokens:
        by[v].append((f1, f2, f3))

    for i, v in enumerate(VOWEL_SET):
        if by[v]:
            mf[3 * i: 3 * i + 3] = np.mean(by[v], axis=0).astype(np.float32)

    allf = np.array([[f1, f2, f3] for _, f1, f2, f3, _ in tokens], np.float32) if tokens else np.zeros((0, 3), np.float32)
    f0s = np.array([f0 for *_, f0 in tokens], np.float32)
    f0s = f0s[np.isfinite(f0s)]

    ltfd = (np.concatenate([allf.mean(0), allf.std(0)]).astype(np.float32)
            if len(allf) else np.full(6, np.nan, np.float32))
    ltf0 = (np.array([f0s.mean(), f0s.std()], np.float32)
            if len(f0s) else np.full(2, np.nan, np.float32))

    return np.concatenate([mf, ltfd, ltf0]).astype(np.float32), len(tokens)


def extract_for_speaker(
    audio_paths: list[str | None], sess=None, extractor=None,
    model_path=None, tokens_path=None, cap: int = 192000,
) -> tuple[np.ndarray, int]:
    """
    Extract formant features for one speaker from multiple audio files.
    Returns (feat[29], total_n_vowels).
    """
    if sess is None:
        sess = make_session(model_path)
    if extractor is None:
        extractor = make_fbank_extractor()
    vocab = load_vocab(tokens_path)

    all_tokens = []
    for p in audio_paths:
        if p is None:
            continue
        try:
            audio = load_audio(p)
        except Exception:
            continue
        audio = audio[:cap]
        tokens = utterance_vowel_formants(audio, sess, extractor, vocab)
        all_tokens.extend(tokens)

    return speaker_formant_vector(all_tokens)
