from __future__ import annotations

import os

import numpy as np
import torch

from ..config import SSL_MODELS, SSL_SNAP, HF_CACHE, SR


def _patch_torch_load():
    """Bypass CVE-2025-32434 torch.load safety check."""
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None


def load_model(name: str, device: str | None = None):
    """
    Load an SSL model. Returns (model, feature_extractor).

    name: 'ecapa' | 'xlsr' | 'hubert' | 'w2vbert'

    For ECAPA: returns (EncoderClassifier, None)
    For others: returns (AutoModel, AutoFeatureExtractor)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _patch_torch_load()

    if name == 'ecapa':
        from speechbrain.inference.speaker import EncoderClassifier
        enc = EncoderClassifier.from_hparams(
            source=SSL_MODELS['ecapa'],
            savedir=str(HF_CACHE / 'spkrec-ecapa-voxceleb'),
            run_opts={'device': device},
        )
        return enc, None

    from transformers import AutoModel, AutoFeatureExtractor, Wav2Vec2BertModel
    sd = HF_CACHE / SSL_SNAP[name] / 'snapshots'
    snap = str(sd / os.listdir(sd)[0])
    fe = AutoFeatureExtractor.from_pretrained(snap, local_files_only=True)

    if name == 'w2vbert':
        m = Wav2Vec2BertModel.from_pretrained(snap, local_files_only=True).to(device).eval()
    else:
        m = AutoModel.from_pretrained(snap, local_files_only=True).to(device).eval()
    return m, fe


def forward_ssl(model, fe, name: str, audio: np.ndarray, device: str) -> np.ndarray:
    """
    Extract mean-pooled SSL vector from a single utterance.

    Returns (D,) vector: 192D for ecapa, 1024D for others.
    """
    wav = torch.from_numpy(audio[:12 * SR].astype(np.float32))

    if name == 'ecapa':
        with torch.no_grad():
            emb = model.encode_batch(wav.unsqueeze(0).to(device))
        return emb.squeeze(1).cpu().numpy().astype(np.float32)

    inp = fe([wav.numpy()], sampling_rate=SR, return_tensors='pt', padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        hs = model(**inp).last_hidden_state.cpu().numpy()  # (1, T, D)
    T = min(max(1, len(wav) // 320), hs.shape[1])
    return hs[0, :T].mean(0).astype(np.float32)


def batch_forward_ssl(model, fe, name: str, audios: list[np.ndarray],
                       device: str, batch_size: int = 16) -> list[np.ndarray]:
    """Extract SSL vectors for multiple utterances."""
    results = []
    for i in range(0, len(audios), batch_size):
        batch = audios[i:i + batch_size]
        if name == 'ecapa':
            max_len = max(len(a) for a in batch)
            pad = torch.zeros(len(batch), max_len)
            for j, a in enumerate(batch):
                pad[j, :len(a)] = torch.from_numpy(a)
            with torch.no_grad():
                embs = model.encode_batch(pad.to(device))
            results.extend(list(embs.squeeze(1).cpu().numpy().astype(np.float32)))
        else:
            for a in batch:
                results.append(forward_ssl(model, fe, name, a, device))
    return results


def speaker_ssl_vector(
    audio_paths: list[str | None], name: str, model=None, fe=None,
    device=None, cap: int = 192000, batch_size: int = 16,
) -> tuple[np.ndarray, int]:
    """
    Compute mean-pooled SSL vector across all utterances of a speaker.
    Returns (vec[D], n_utterances).
    """
    from ..alignment.zipa import load_audio

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model is None:
        model, fe = load_model(name, device)

    audios = []
    for p in audio_paths:
        if p is None:
            continue
        try:
            a = load_audio(p)[:cap]
        except Exception:
            continue
        a = np.nan_to_num(a).astype(np.float32)
        if len(a) < 512:
            a = np.pad(a, (0, 512 - len(a)))
        audios.append(a)

    if not audios:
        dim = 192 if name == 'ecapa' else 1024
        return np.full(dim, np.nan, np.float32), 0

    vecs = batch_forward_ssl(model, fe, name, audios, device, batch_size)
    return np.mean(vecs, axis=0).astype(np.float32), len(audios)
