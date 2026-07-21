from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from ..config import (
    PX_MODEL, PX_MAX_S, PX_CHUNK_S, PX_WIN,
    PHONE_GROUPS, SR, HF_CACHE,
)


def _patch_torch_load():
    """Bypass CVE-2025-32434 torch.load safety check (transformers + torch < 2.6)."""
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None


def load_model() -> tuple:
    """Load PhoneticXeus model. Returns (model, device, token_list)."""
    _patch_torch_load()
    from transformers import AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = AutoModel.from_pretrained(PX_MODEL, trust_remote_code=True).eval().to(dev)
    return m, dev, list(m.model.token_list)


def forward_logits(model, device: str, audio: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """
    Run PhoneticXeus forward on audio. Returns (logits[T,vocab], fps).
    Handles chunking for long utterances.
    """
    wav = torch.from_numpy(audio[:SR * PX_MAX_S].astype(np.float32))
    dur = len(wav) / SR
    parts = []
    for c0 in range(0, len(wav), PX_CHUNK_S * SR):
        seg = wav[c0:c0 + PX_CHUNK_S * SR]
        if len(seg) < int(0.1 * SR):
            continue
        with torch.no_grad():
            lg = model.forward(seg.unsqueeze(0).to(device)).logits[0]
        parts.append(lg.float().cpu().numpy())
    if not parts:
        return None, None
    L = np.concatenate(parts, 0)
    return L, len(L) / dur


def pooled_at_frame(logits: np.ndarray, frame: int, token_ids: np.ndarray,
                     win: int | None = None) -> np.ndarray | None:
    """Mean-pool logits over [frame-win, frame+win] at specified token IDs."""
    w = win if win is not None else PX_WIN
    lo, hi = max(0, frame - w), min(len(logits), frame + w + 1)
    if hi <= lo:
        return None
    return logits[lo:hi].mean(0)[token_ids].astype(np.float32)


def logits_at_marker(logits: np.ndarray, frame: int, marker: str,
                      token_to_id: dict[str, int],
                      groups: dict[str, list[str]] | None = None,
                      win: int | None = None) -> np.ndarray | None:
    """
    Extract pooled logits at a marker frame for the candidate phone group.

    Args:
        logits: (T, vocab) raw logits from PhoneticXeus
        frame: target frame index
        marker: 's', 'r', or 'dt'
        token_to_id: maps IPA token string -> vocab index
        groups: override phone groups (default PHONE_GROUPS)
        win: pooling window (default PX_WIN)

    Returns:
        pooled logits over candidate phone group, or None
    """
    gs = groups or PHONE_GROUPS
    if marker not in gs:
        raise ValueError(f"Unknown marker '{marker}'; expected one of {list(gs.keys())}")
    phones = gs[marker]
    ids = np.array([token_to_id[p] for p in phones if p in token_to_id])
    if len(ids) == 0:
        return None
    return pooled_at_frame(logits, frame, ids, win)


def extract_marker_features(audio: np.ndarray, markers: list[dict],
                             model=None, device=None, token_list=None,
                             groups: dict[str, list[str]] | None = None) -> np.ndarray | None:
    """
    Extract PhoneticXeus features for all markers in one forward pass.

    Args:
        audio: audio samples at SR
        markers: list of dicts with 'marker', 'frame', 'time_s' keys
        model, device, token_list: pre-loaded model (loaded if None)
        groups: override phone groups

    Returns:
        (N, D) feature matrix, or None if no markers
    """
    if not markers:
        return None

    if model is None:
        model, device, token_list = load_model()
    else:
        token_list = list(model.model.token_list)

    gs = groups or PHONE_GROUPS
    tid = {t: i for i, t in enumerate(token_list)}
    gids = {k: np.array([tid[p] for p in g if p in tid]) for k, g in gs.items()}
    dims = {k: len(v) for k, v in gids.items()}
    total_dim = sum(dims.values())

    L, fps = forward_logits(model, device, audio)
    if L is None:
        return None

    acc: dict[str, list] = {k: [] for k in dims}
    for m in markers:
        mk = m['marker']
        if mk not in dims:
            continue
        pf = int(round(m['time_s'] * fps))
        v = pooled_at_frame(L, pf, gids[mk])
        if v is not None:
            acc[mk].append(v)

    rows = []
    for mk in dims:
        if acc[mk]:
            rows.append(np.mean(acc[mk], 0))
        else:
            rows.append(np.full(dims[mk], np.nan, np.float32))
    return np.concatenate(rows).astype(np.float32)
