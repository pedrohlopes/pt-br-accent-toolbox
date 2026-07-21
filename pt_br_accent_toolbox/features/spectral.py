from __future__ import annotations

import numpy as np

N_FFT = 4096
F_LO, F_HI = 500, 8000


def spectral_moments(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Compute 6D spectral moments over a short audio clip.

    Returns: [centroid, spread, skewness, kurtosis, peak_freq, log_band_ratio]
    """
    f = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    n = min(len(audio), N_FFT)
    sig = audio[:n] * np.hanning(n).astype(np.float32)
    if n < N_FFT:
        sig = np.pad(sig, (0, N_FFT - n))
    P = np.abs(np.fft.rfft(sig)) ** 2
    band = (f >= F_LO) & (f <= F_HI)
    Pb, fb = P[band], f[band]
    Z = Pb.sum()
    if Z <= 0:
        return np.zeros(6, dtype=np.float32)
    p = Pb / Z
    m1 = float((fb * p).sum())
    m2 = float(((fb - m1) ** 2 * p).sum())
    sd = np.sqrt(max(m2, 1e-9))
    m3 = float((((fb - m1) / sd) ** 3 * p).sum())
    m4 = float((((fb - m1) / sd) ** 4 * p).sum()) - 3.0
    pk = float(fb[np.argmax(Pb)])
    lo_b = P[(f >= 2000) & (f <= 4000)].sum()
    hi_b = P[(f >= 5000) & (f <= 8000)].sum()
    log_br = float(np.log10((hi_b + 1e-12) / (lo_b + 1e-12)))
    return np.array([m1, m2, m3, m4, pk, log_br], dtype=np.float32)
