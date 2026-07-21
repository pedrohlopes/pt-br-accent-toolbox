from __future__ import annotations

import numpy as np
from . import spectral

N_FFT = spectral.N_FFT
F_LO, F_HI = spectral.F_LO, spectral.F_HI

WIN = 2


def windowed_probs(log_probs: np.ndarray, frame: int, phone_ids: np.ndarray,
                    win: int = WIN) -> np.ndarray:
    """
    Mean-pool softmax probabilities at [frame-win, frame+win] for given phone IDs.
    """
    T = log_probs.shape[0]
    z = log_probs.max(axis=-1, keepdims=True)
    probs = np.exp(log_probs - z)
    probs /= probs.sum(axis=-1, keepdims=True)
    lo = max(0, frame - win)
    hi = min(T, frame + win + 1)
    return probs[lo:hi, phone_ids].mean(axis=0).astype(np.float32)


def clip_features(audio_clip: np.ndarray, log_probs: np.ndarray,
                   frame: int, phone_ids: np.ndarray,
                   sr: int = 16000, win: int = WIN) -> np.ndarray:
    """
    Combined spectral moments + ZIPA logits for a single marker clip.
    Returns (6 + len(phone_ids))-dim vector.
    """
    sm = spectral.spectral_moments(audio_clip, sr)
    zl = windowed_probs(log_probs, frame, phone_ids, win)
    return np.concatenate([sm, zl])


def extract_for_markers(audio: np.ndarray, markers: list[dict],
                         log_probs: np.ndarray, token_to_id: dict[str, int],
                         sr: int = 16000, clip_half: int = 320,
                         groups: dict[str, list[str]] | None = None) -> np.ndarray:
    """
    Extract [spec6 | zipa_group] features for each marker.
    Returns (N, D) matrix where D depends on phone group size.
    """
    from ..config import PHONE_GROUPS
    gs = groups or PHONE_GROUPS

    rows = []
    for m in markers:
        mk = m['marker']
        phones = gs.get(mk, [])
        ids = np.array([token_to_id[p] for p in phones if p in token_to_id])
        if len(ids) == 0:
            continue

        center = int(m['time_s'] * sr)
        lo = max(0, center - clip_half)
        hi = min(len(audio), center + clip_half)
        clip = audio[lo:hi]
        row = clip_features(clip, log_probs, m['frame'], ids, sr)
        rows.append(row)

    if not rows:
        return np.zeros((0, 0), np.float32)
    return np.stack(rows)
