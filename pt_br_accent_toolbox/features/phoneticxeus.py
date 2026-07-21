from __future__ import annotations

import numpy as np
import torch

from ..config import PHONE_GROUPS, SR
from ..alignment.phoneticxeus import (
    load_model as load_px_model, forward_logits, pooled_at_frame,
)


def extract_for_markers(audio: np.ndarray, markers: list[dict],
                         model=None, device=None, token_list=None,
                         groups: dict[str, list[str]] | None = None) -> np.ndarray:
    """
    Extract PhoneticXeus group-logit features for each marker position.
    Returns (N, D) matrix, D = sum of group sizes.

    Markers with no valid logits get NaN.
    """
    if not markers:
        return np.zeros((0, 0), np.float32)

    if model is None:
        model, device, token_list = load_px_model()
    else:
        token_list = list(model.model.token_list)

    gs = groups or PHONE_GROUPS
    tid = {t: i for i, t in enumerate(token_list)}
    gids = {k: np.array([tid[p] for p in g if p in tid]) for k, g in gs.items()}
    gnm = {k: [p for p in g if p in tid] for k, g in gs.items()}
    dims = {k: len(v) for k, v in gids.items()}
    total_dim = sum(dims.values())

    L, fps = forward_logits(model, device, audio)
    if L is None:
        return np.zeros((len(markers), total_dim), np.float32)

    rows = []
    for m in markers:
        row = np.full(total_dim, np.nan, np.float32)
        mk = m['marker']
        if mk not in dims:
            rows.append(row)
            continue
        pf = int(round(m['time_s'] * fps))
        v = pooled_at_frame(L, pf, gids[mk])
        if v is not None:
            row[:] = v
        rows.append(row)

    return np.stack(rows)


def speaker_vector(audio_files: list, markers_per_file: list[list[dict]],
                    model=None, device=None, token_list=None,
                    groups: dict[str, list[str]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate PX features across all utterances of a speaker.
    Returns (speaker_vec, counts) where counts is (N, 3) [s, r, dt].
    """
    if model is None:
        model, device, token_list = load_px_model()

    gs = groups or PHONE_GROUPS
    tid = {t: i for i, t in enumerate(token_list)}
    gids = {k: np.array([tid[p] for p in g if p in tid]) for k, g in gs.items()}
    dims = {k: len(v) for k, v in gids.items()}

    acc: dict[str, list] = {k: [] for k in dims}

    for audio_path, markers in zip(audio_files, markers_per_file):
        from ..alignment.zipa import load_audio
        audio = load_audio(audio_path)
        L, fps = forward_logits(model, device, audio)
        if L is None:
            continue

        for m in markers:
            mk = m['marker']
            if mk not in dims:
                continue
            pf = int(round(m['time_s'] * fps))
            v = pooled_at_frame(L, pf, gids[mk])
            if v is not None:
                acc[mk].append(v)

    rows = []
    counts = []
    for mk in dims:
        if acc[mk]:
            rows.append(np.mean(acc[mk], 0))
        else:
            rows.append(np.full(dims[mk], np.nan, np.float32))
        counts.append(len(acc[mk]))

    return np.concatenate(rows).astype(np.float32), np.array(counts, np.int32)
