from __future__ import annotations

import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
from lhotse.features.kaldi.extractors import Fbank, FbankConfig

from .config import ZIPA_MODEL, ZIPA_TOKENS, SR, FRAME_MS


def load_vocab(tokens_path: Path | str | None = None) -> dict[int, str]:
    """Load ZIPA vocab: id -> IPA token string."""
    p = Path(tokens_path or ZIPA_TOKENS)
    tok2id: dict[int, str] = {}
    for line in open(p, encoding='utf-8'):
        parts = line.strip().split()
        if len(parts) >= 2:
            tok2id[int(parts[1])] = parts[0]
    return tok2id


def load_audio(path: Path | str) -> np.ndarray:
    """Load audio as mono float32 at SR. Resamples if needed."""
    try:
        a, sr = sf.read(str(path))
        if a.ndim > 1:
            a = a[:, 0]
        a = a.astype(np.float32)
    except Exception:
        import librosa
        a, sr = librosa.load(str(path), sr=None, mono=True)
        a = a.astype(np.float32)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a


def make_fbank_extractor() -> Fbank:
    """Return Lhotse Fbank extractor matching ZIPA training config."""
    return Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))


def make_session(model_path: Path | str | None = None) -> ort.InferenceSession:
    """Create ONNX session for ZIPA model."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(str(model_path or ZIPA_MODEL), providers=providers)


# ── frame-level inference ────────────────────────────────────────────────────

def utterance_logprobs(audio: np.ndarray, sess: ort.InferenceSession,
                       extractor: Fbank) -> np.ndarray:
    """Run ZIPA CTC on a single utterance. Returns (T, V) log-probabilities."""
    a = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    feat = extractor.extract_batch([a], sampling_rate=SR)[0].unsqueeze(0).numpy()
    feat_lens = np.array([feat.shape[1]], dtype=np.int64)
    return sess.run(None, {'x': feat, 'x_lens': feat_lens})[0][0]


def ctc_greedy_decode(log_probs: np.ndarray, vocab: dict[int, str],
                       blank_id: int = 0) -> list[str]:
    """Greedy CTC decode: collapse blanks and repeats."""
    preds = np.argmax(log_probs, axis=-1)
    decoded: list[str] = []
    prev = -1
    for idx in preds:
        idx = int(idx)
        if idx != blank_id and idx != prev:
            token = vocab.get(idx, '')
            if token and token not in ('<blk>', '<sos/eos>', '<unk>'):
                decoded.append(token)
        prev = idx
    return decoded


def get_spikes(log_probs: np.ndarray, vocab: dict[int, str],
               blank_id: int = 0) -> list[tuple[float, str]]:
    """Return list of (frame_idx, ipa_token) at CTC change points."""
    preds = np.argmax(log_probs, axis=-1)
    spikes: list[tuple[float, str]] = []
    prev = -1
    for t, idx in enumerate(preds):
        idx = int(idx)
        if idx == blank_id:
            prev = -1
            continue
        if idx == prev:
            continue
        prev = idx
        spikes.append((t, vocab.get(idx, '?')))
    return spikes


# ── phonological marker detection ────────────────────────────────────────────

VOWELS = set('aeiouɐɛɔɪʊəɨɯ')
WORD_BD = '▁'

S_CODA_SRC = {'s', 'ʃ'}
R_CODA_SRC = {'ɾ', 'ʁ', 'ʀ', 'ħ', 'χ', 'r', 'h', 'x'}
TD_PHONES = {'t', 'd'}
FRONT_NEXT = {'i', 'ɪ', 'ʃ', 'ʒ', 'j'}


def is_vowel(t: str) -> bool:
    return bool(t) and t[0] in VOWELS


def is_coda_s(phone: str, next_phone: str) -> bool:
    return phone in S_CODA_SRC and (next_phone == '' or next_phone == WORD_BD or not is_vowel(next_phone))


def is_coda_r(phone: str, next_phone: str) -> bool:
    return phone in R_CODA_SRC and (next_phone == '' or next_phone == WORD_BD or not is_vowel(next_phone))


def is_dt(phone: str, next_phone: str) -> bool:
    return phone in TD_PHONES and next_phone in FRONT_NEXT


# ── marker frame extraction ─────────────────────────────────────────────────

def extract_marker_frames(audio: np.ndarray, sess: ort.InferenceSession | None = None,
                          extractor: Fbank | None = None,
                          model_path=None, tokens_path=None) -> list[dict]:
    """
    Run ZIPA alignment on audio, return marker frames with metadata.

    Returns list of dicts:
      {'marker': 's'|'r'|'dt', 'frame': int, 'time_s': float,
       'phone': str, 'next_phone': str}
    """
    if sess is None:
        sess = make_session(model_path)
    if extractor is None:
        extractor = make_fbank_extractor()
    vocab = load_vocab(tokens_path)
    lp = utterance_logprobs(audio, sess, extractor)
    spikes = get_spikes(lp, vocab)

    markers: list[dict] = []
    for k, (t, ph) in enumerate(spikes):
        nx = spikes[k + 1][1] if k + 1 < len(spikes) else ''
        time_s = t * FRAME_MS / 1000.0
        if is_coda_s(ph, nx):
            markers.append({'marker': 's', 'frame': t, 'time_s': time_s,
                            'phone': ph, 'next_phone': nx})
        if is_coda_r(ph, nx):
            markers.append({'marker': 'r', 'frame': t, 'time_s': time_s,
                            'phone': ph, 'next_phone': nx})
        if is_dt(ph, nx):
            markers.append({'marker': 'dt', 'frame': t, 'time_s': time_s,
                            'phone': ph, 'next_phone': nx})
    return markers


def extract_phoneme_sequence(audio: np.ndarray, sess: ort.InferenceSession | None = None,
                              extractor: Fbank | None = None,
                              model_path=None, tokens_path=None) -> list[str]:
    """Full CTC-greedy decode of utterance."""
    if sess is None:
        sess = make_session(model_path)
    if extractor is None:
        extractor = make_fbank_extractor()
    vocab = load_vocab(tokens_path)
    lp = utterance_logprobs(audio, sess, extractor)
    return ctc_greedy_decode(lp, vocab)
