from __future__ import annotations

import numpy as np


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Extract MFCC features from audio.
    Returns (n_mfcc, T) — transpose to (T, n_mfcc) for standard usage.
    """
    import librosa
    return librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def speaker_mfcc_mean(
    audio_paths: list[str | None], sr: int = 16000,
    n_mfcc: int = 13, n_fft: int = 400, hop_length: int = 160,
) -> np.ndarray:
    """
    Compute mean MFCC vector across all utterances of a speaker.
    Returns (n_mfcc,) vector.
    """
    from ..alignment.zipa import load_audio
    chunks = []
    for p in audio_paths:
        if p is None:
            continue
        try:
            audio = load_audio(p)
        except Exception:
            continue
        mfcc = extract_mfcc(audio, sr, n_mfcc, n_fft, hop_length)
        chunks.append(mfcc.mean(axis=1))
    if not chunks:
        return np.zeros(n_mfcc, np.float32)
    return np.mean(chunks, axis=0)
