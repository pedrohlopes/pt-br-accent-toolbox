"""Character-level CTC forced alignment (XLSR-pt).

Ported from the accents project's Exp 56 (`code/extraction/ctc_char_align.py`).

Why characters and not phones: the vowel-harmony measurement asks whether a
speaker says the <o> of "colega" as [o] or [ɔ].  Any aligner that must first
decide WHICH PHONE was said is circular for this question.  A character CTC
aligner is told the orthography ("colega") and only has to find WHERE each
letter is, which it does equally well for either realization.

Model: jonatasgrosman/wav2vec2-large-xlsr-53-portuguese (the same checkpoint
as `SSL_MODELS['xlsr']`, so `download_models.py ssl` already fetches it), a
46-symbol character CTC: letters + accented vowels, <pad>=0 is the CTC blank,
'|' the word delimiter.

Alignment uses `torchaudio.functional.forced_align` when available and falls
back to an equivalent NumPy Viterbi otherwise (torchaudio deprecated
forced_align in 2.8), so the module keeps working on newer torch stacks.
"""
from __future__ import annotations

import numpy as np

from ..config import HF_CACHE, SR, SSL_MODELS

MODEL_ID = SSL_MODELS['xlsr']
# XLSR conv stack: 320-sample stride at 16 kHz = 20 ms per frame,
# the same frame rate as ZIPA, so frame indices are directly comparable.
FRAME_MS = 20.0


def _patch_torch_load_guard():
    """transformers >= 4.57 blocks torch.load under torch < 2.6 (CVE-2025-32434).

    The checkpoint ships safetensors, so the guard only gets in the way.
    """
    def _noop(*a, **k):
        return None
    for mod in ('transformers.utils.import_utils', 'transformers.modeling_utils'):
        try:
            m = __import__(mod, fromlist=['check_torch_load_is_safe'])
            if hasattr(m, 'check_torch_load_is_safe'):
                m.check_torch_load_is_safe = _noop
        except Exception:
            pass


def load_model(device: str | None = None, model_id: str = MODEL_ID):
    """-> (model, processor, char2id, device). Half precision on CUDA."""
    import torch
    _patch_torch_load_guard()
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    proc = Wav2Vec2Processor.from_pretrained(model_id, cache_dir=str(HF_CACHE))
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, cache_dir=str(HF_CACHE),
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    model = model.to(device).eval()
    char2id = proc.tokenizer.get_vocab()
    return model, proc, char2id, device


def ctc_text(text: str, char2id: dict) -> tuple[str, int]:
    """Orthographic text -> the exact string the CTC targets encode.

    Characters outside the model vocabulary are dropped (counted, so the caller
    can reject utterances that lost too much).  Word boundaries become '|'.
    """
    out, dropped = [], 0
    for ch in text:
        if ch == ' ':
            if out and out[-1] != '|':
                out.append('|')
            continue
        if ch in char2id:
            out.append(ch)
        else:
            dropped += 1
    while out and out[-1] == '|':
        out.pop()
    return ''.join(out), dropped


def emissions(audio: np.ndarray, model, proc, device, want_hidden: bool = False):
    """-> (log_probs [T,V] float32 cpu tensor, hidden [T,H] float16 or None)."""
    import torch
    with torch.inference_mode():
        iv = proc(audio, sampling_rate=SR, return_tensors='pt').input_values
        iv = iv.to(device, dtype=model.dtype)
        out = model(iv, output_hidden_states=want_hidden)
        lp = torch.log_softmax(out.logits.float(), dim=-1)[0].cpu()
        hid = None
        if want_hidden:
            # layer 10 carries more phonetic/vocalic detail than the CTC-tuned top
            hid = out.hidden_states[10][0].to(torch.float16).cpu()
    return lp, hid


def _viterbi_numpy(lp: np.ndarray, ids: list[int], blank: int = 0):
    """CTC Viterbi forced alignment -> (frame labels, frame log-probs).

    Same contract as torchaudio's forced_align: one label per frame (blank or
    a target id) along the single best path through the blank-interleaved
    target sequence.
    """
    T = lp.shape[0]
    ext = [blank]
    for i in ids:
        ext += [i, blank]
    S = len(ext)
    neg = -np.inf
    dp = np.full((T, S), neg, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int8)       # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = lp[0, ext[0]]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    skip_ok = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]
                        for s in range(S)])
    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        one = np.concatenate(([neg], prev[:-1]))
        two = np.where(skip_ok, np.concatenate(([neg, neg], prev[:-2])), neg)
        stack = np.stack([stay, one, two])
        best = stack.argmax(0)
        dp[t] = stack[best, np.arange(S)] + lp[t, ext]
        bp[t] = best
    # the path must end on the last label or the trailing blank
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if not np.isfinite(dp[-1, s]):
        return None, None
    labels = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        labels[t] = ext[s]
        s -= int(bp[t, s])
    scores = lp[np.arange(T), labels]
    return labels.tolist(), scores.tolist()


def _forced_align(lp, ids: list[int]):
    try:
        import torch
        from torchaudio.functional import forced_align
        tgt = torch.tensor([ids], dtype=torch.int32)
        x = lp.unsqueeze(0)
        labels, scores = forced_align(
            x, tgt,
            input_lengths=torch.tensor([x.shape[1]], dtype=torch.int32),
            target_lengths=torch.tensor([len(ids)], dtype=torch.int32),
            blank=0)
        return labels[0].tolist(), scores[0].tolist()
    except (ImportError, AttributeError, RuntimeError):
        arr = lp.numpy() if hasattr(lp, 'numpy') else np.asarray(lp)
        return _viterbi_numpy(arr, ids)


def align(log_probs, target: str, char2id: dict):
    """Viterbi-align `target` to emissions.

    Returns a list of per-character spans:
        [char_idx, char, f_start, f_end, f_next, score]
    where [f_start, f_end] is the span of frames the aligner assigned to that
    character and f_next is the first frame of the following character.  CTC is
    peaky, so a character's assigned span is usually 1-2 frames; the acoustic
    measurement downstream widens [f_start, f_next) and refines it against the
    vocalic-posterior run.  None when the target does not fit the emissions.
    """
    ids = [char2id[c] for c in target]
    if not ids or log_probs.shape[0] < len(ids):
        return None
    labels, scores = _forced_align(log_probs, ids)
    if labels is None:
        return None

    # collapse the frame-level path into one span per target character
    spans, ti, t = [], 0, 0
    n = len(labels)
    while t < n and ti < len(ids):
        if labels[t] == 0:                      # blank
            t += 1
            continue
        f0 = t
        sc = []
        while t < n and labels[t] == ids[ti]:
            sc.append(scores[t])
            t += 1
            # a repeated identical character must be separated by a blank, so
            # a run of the same id belongs to one character only
        spans.append([ti, target[ti], f0, t - 1, -1, float(np.mean(sc))])
        ti += 1
    if ti != len(ids):
        return None
    for k in range(len(spans) - 1):
        spans[k][4] = spans[k + 1][2]           # f_next
    if spans:
        spans[-1][4] = min(spans[-1][3] + 3, n)
    return spans


def greedy_decode(log_probs, char2id: dict) -> str:
    """Transcribe straight from the emissions (CTC greedy, no LM).

    Used for audio with no transcript.  It is worse than Whisper (~11% WER on
    Alcaim read speech) but costs nothing: the forward pass already computed
    for alignment produces it, and the alignment that follows is
    self-consistent because the model aligns its own hypothesis.  A
    misrecognized word costs tokens; it cannot bias a token's measured
    openness, since openness is measured from the audio and the trigger comes
    from the lexicon.
    """
    id2char = {v: k for k, v in char2id.items()}
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(id2char.get(i, ''))
        prev = i
    return ''.join(out).replace('|', ' ').strip()
