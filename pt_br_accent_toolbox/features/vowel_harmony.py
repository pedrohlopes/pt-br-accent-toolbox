"""
Pretonic mid-vowel lowering ("vowel harmony") — a continuous per-speaker score.

Ported from the accents project's Exp 56 (1,153 speakers, 9 corpora, ~735k
vowel tokens).  In the North/Northeast of Brazil a pretonic /e, o/ opens to
[ɛ, ɔ] when the stressed vowel of the same word is open (c[ɔ]léga,
m[ɔ]léque), while Southeastern speakers keep [e, o].  This module measures that
opening acoustically, with no vowel-quality transcription and no labels.

Measurement chain — four independent sources of truth:

    text + espeak-ng G2P   WHERE the pretonic vowel is, the LEXICAL trigger and
                           stress (never how the vowel was realized)
    XLSR character CTC     the time span of the vowel LETTER
    XLSR vocalic posterior refinement of that span to the vocalic run
    Praat / parselmouth    F1 — the openness measurement itself

ZIPA is deliberately NOT used: it cannot separate [e] from [ɛ] (per-speaker
F1 gap of ZIPA-labelled [ɛ]-[e] is only +13..20 Hz), and using a phone
recognizer to locate a lowered /e/ would be circular.  The character aligner
recovers a +114 Hz [e]->[ɛ] gap (d = 1.21).

Normalization — every token is expressed against the speaker's OWN
lexically-stressed anchors:

    h = (F1_token − F1̄[e]stressed) / (F1̄[ɛ]stressed − F1̄[e]stressed)

(and [o]/[ɔ] for <o>).  h = 0 is "as closed as their own [e]", h = 1 "as open
as their own [ɛ]".  The same anchors are a free QA gate (`anchor_ok`).

Speaker indices:
    OI = mean h before an open stressed vowel (T_low)  — openness, the primary
         score (NE 0.646 vs SE 0.357, d = +1.33 in Exp 56)
    HI = OI − mean h before a close-mid stressed vowel  — harmony contrast,
         free of the general pretonic-reduction offset

Exp 56 found a CONTINUUM, not two classes (ΔBIC = −42 for a split, one KDE
peak), so this is a scalar marker, unlike the categorical s/r/dt markers.

Typical use:

    from pt_br_accent_toolbox.features.vowel_harmony import HarmonyMeasurer
    hm = HarmonyMeasurer()                       # loads XLSR once
    vec, meta = hm.speaker_vector(paths, texts)  # texts optional (greedy ASR)
    meta['descriptors']['OI'], meta['descriptors']['gate_ok']

Requirements: `espeak-ng` on PATH, `transformers`, and `torchaudio` (optional —
a NumPy Viterbi is used when torchaudio's forced_align is unavailable).
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import SR

# ── constants (Exp 56) ───────────────────────────────────────────────────────

MIN_DUR_MS, MAX_DUR_MS = 30.0, 300.0
MAX_GROW = 6               # frames the span may grow each way (120 ms)
VOWEL_POST_THR = 0.5       # vocalic-posterior threshold for span growth

# anchor QA: an [e]-[ɛ] contrast in BP is ~100-200 Hz; failures look like the
# 13-20 Hz that ZIPA-labelled vowels give
MIN_GAP_HZ, MIN_D, MIN_GAP_AE = 60.0, 0.7, 150.0
MIN_ANCHOR_N = 8
MIN_LOW, MIN_MID = 15, 8   # token gates for a usable speaker
H_CLIP = (-1.0, 2.0)
ANCHOR_VOWELS = ['a', 'e', 'ɛ', 'o', 'ɔ', 'i', 'u']

FEATURE_NAMES = ['OI', 'HI', 'open_rate_low', 'OR_delta', 'grad_slope', 'RI']
FEAT_DIM = len(FEATURE_NAMES)

_VOWEL_LETTERS_CTC = 'aeiouáéêíóôúâãõà'


# ── audio front-end ──────────────────────────────────────────────────────────

def normalize_audio(a: np.ndarray, top_db: float = 30.0,
                    target_rms: float = 0.05) -> np.ndarray:
    """Silence-trim + RMS-normalize (the Exp 52 channel front-end).

    Makes formants comparable across corpora recorded under different gains.
    """
    import librosa
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    if len(a) < 64:
        return a
    try:
        t, _ = librosa.effects.trim(a, top_db=top_db)
        if len(t) >= 64:
            a = t
    except Exception:
        pass
    a = a * (target_rms / (float(np.sqrt(np.mean(a ** 2))) + 1e-9))
    pk = float(np.max(np.abs(a)))
    if pk > 0.99:
        a = a * (0.99 / pk)
    return a.astype(np.float32)


def load_audio_norm(path, normalize: bool = True) -> np.ndarray:
    import librosa
    a, _ = librosa.load(str(path), sr=SR, mono=True)
    return normalize_audio(a) if normalize else a.astype(np.float32)


# ── token measurement ────────────────────────────────────────────────────────

def vocalic_posterior(log_probs, char2id: dict) -> np.ndarray:
    """Per-frame vocalic mass from the character CTC (letter identity ignored)."""
    import torch
    cols = [i for c, i in char2id.items() if c in _VOWEL_LETTERS_CTC]
    return torch.softmax(log_probs, dim=-1)[:, cols].sum(-1).numpy()


def refine_span(f_start, f_next, vmass, lo_bound=None, hi_bound=None):
    """Grow the vowel span outward from the CTC spike along the vocalic run.

    CTC is peaky (1-2 frames per letter) — reliable as a POSITION, too short to
    measure on.  Growth is bounded by the neighbouring characters' spikes: in
    "colega" the pretonic <o> is separated from the stressed <e> only by /l/,
    and a liquid keeps the vocalic posterior high, so an unbounded span would
    grow into the stressed vowel and manufacture the very effect under test.
    """
    n = len(vmass)
    seed = min(max(int(f_start), 0), n - 1)
    lo, hi = seed, max(int(f_next), seed + 1)
    lo_lim = max(0 if lo_bound is None else max(0, lo_bound), seed - MAX_GROW)
    hi_lim = min(n if hi_bound is None else min(n, hi_bound), seed + MAX_GROW + 1)
    grew = False
    while lo - 1 >= lo_lim and vmass[lo - 1] > VOWEL_POST_THR:
        lo -= 1
        grew = True
    while hi < hi_lim and vmass[min(hi, n - 1)] > VOWEL_POST_THR:
        hi += 1
        grew = True
    hi = min(hi, n)
    if hi - lo < 2:
        hi = min(n, lo + 2)
    return lo, hi, ('vocalic' if grew else 'ctc')


def measure_utterance(audio, text, lex, model, proc, char2id, device,
                      utt_id: str = '') -> tuple[list[dict], dict]:
    """One utterance -> (token rows, QA dict).

    Each row is a vowel nucleus with role 'pretonic' (target), 'pre2', or
    'anchor' (stressed reference vowel), its trigger context, the aligned
    segment and F1/F2/F3.  An empty `text` is transcribed by greedy CTC from
    the same forward pass.
    """
    import parselmouth
    from ..alignment.ctc_char import ctc_text, emissions, align, greedy_decode, FRAME_MS
    from ..alignment.g2p_stress import normalize_text

    qa = {'utt': utt_id, 'n_tokens': 0, 'ctc_ok': 0, 'dropped_chars': 0,
          'utt_score': np.nan, 'n_words': 0, 'n_words_ok': 0, 'asr': 0}
    if len(audio) < SR // 4:
        return [], qa

    lp, _ = emissions(audio, model, proc, device)
    if not str(text or '').strip():
        text = greedy_decode(lp, char2id)
        qa['asr'] = 1
    norm = normalize_text(text)
    if not norm:
        return [], qa

    words = norm.split()
    ana = lex.analyze(list(set(words)))
    qa['n_words'] = len(words)
    qa['n_words_ok'] = sum(1 for w in words if ana.get(w, {}).get('ok'))

    target, dropped = ctc_text(norm, char2id)
    qa['dropped_chars'] = dropped
    if not target:
        return [], qa

    offsets, pos = [], 0
    for w in words:
        wt, _ = ctc_text(w, char2id)
        idx = target.find(wt, pos)
        offsets.append(idx)
        if idx >= 0:
            pos = idx + len(wt)

    spans = align(lp, target, char2id)
    if spans is None:
        return [], qa
    qa['ctc_ok'] = 1
    qa['utt_score'] = float(np.mean([s[5] for s in spans]))
    vmass = vocalic_posterior(lp, char2id)

    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
    dur_s = snd.get_total_duration()
    try:
        pitch = snd.to_pitch()
        f0v = pitch.selected_array['frequency']
        f0v = f0v[f0v > 0]
        f0_med = float(np.median(f0v)) if len(f0v) else np.nan
    except Exception:
        pitch, f0_med = None, np.nan
    # a higher ceiling for high-pitched voices, else F1/F2 merge on open vowels
    ceiling = 5500.0 if (np.isfinite(f0_med) and f0_med > 170) else 5000.0
    fm = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=ceiling,
                             window_length=0.025, pre_emphasis_from=50.0)

    rows = []
    for wi, w in enumerate(words):
        d = ana.get(w)
        if not d or not d['ok'] or offsets[wi] < 0:
            continue
        for u in d['nuclei']:
            if u['role'] == 'other':
                continue
            ci = offsets[wi] + u['letter_idx']
            if ci >= len(spans) or spans[ci][1] != u['letter']:
                continue
            prev_hi = spans[ci - 1][3] + 1 if ci > 0 else None
            next_lo = spans[ci + 1][2] if ci + 1 < len(spans) else None
            a0, a1, src = refine_span(spans[ci][2], spans[ci][4], vmass,
                                      lo_bound=prev_hi, hi_bound=next_lo)
            t0, t1 = a0 * FRAME_MS / 1000.0, a1 * FRAME_MS / 1000.0
            dms = (t1 - t0) * 1000.0
            if dms < MIN_DUR_MS or dms > MAX_DUR_MS or t1 >= dur_s:
                continue
            mid = 0.5 * (t0 + t1)
            ts = np.linspace(t0 + 0.25 * (t1 - t0), t1 - 0.25 * (t1 - t0), 5)
            f1s = np.array([fm.get_value_at_time(1, t) for t in ts], float)
            if not np.isfinite(f1s).any():
                continue
            f2s = np.array([fm.get_value_at_time(2, t) for t in ts], float)
            f3s = np.array([fm.get_value_at_time(3, t) for t in ts], float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                rows.append({
                    'utt': utt_id, 'word': w, 'letter': u['letter'],
                    'letter_idx': u['letter_idx'], 'role': u['role'],
                    'trigger': u['trigger'], 'stressed_ipa': u['stressed_ipa'],
                    'dist': u['dist'], 'flags': '|'.join(u['flags']),
                    'seg_start_s': t0, 'seg_end_s': t1, 'dur_ms': dms,
                    'seg_src': src, 'ctc_score': spans[ci][5],
                    'f1_mid': fm.get_value_at_time(1, mid),
                    'f2_mid': fm.get_value_at_time(2, mid),
                    'f1_m50': float(np.nanmean(f1s)),
                    'f2_m50': float(np.nanmean(f2s)),
                    'f3_m50': float(np.nanmean(f3s)),
                    'f1_sd50': float(np.nanstd(f1s)),
                    'f0_mid': (pitch.get_value_at_time(mid) if pitch else np.nan),
                    'f0_utt_med': f0_med,
                })
    qa['n_tokens'] = len(rows)
    return rows, qa


# ── speaker normalization + descriptors ──────────────────────────────────────

def _col(rows, key, dtype=float):
    if dtype is float:
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return np.array([r.get(key, '') for r in rows], dtype=object)


def _cohen_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan
    s = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return float((np.mean(y) - np.mean(x)) / s) if s > 0 else np.nan


def ashman_d(x) -> float:
    """Ashman's D of a 2-component GMM; D > 2 = a genuinely bimodal split.

    Reported as a categorical-vs-gradient DIAGNOSTIC.  Exp 56 found that
    feeding it into speaker clustering makes BIC fit noise, so it is not part
    of the feature vector.
    """
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    gm = GaussianMixture(2, covariance_type='full', random_state=42, n_init=5)
    gm.fit(x.reshape(-1, 1))
    if min(gm.weights_) < 0.10:          # a mode holding <10% is noise
        return np.nan
    m = gm.means_.ravel()
    s = np.sqrt(gm.covariances_.ravel())
    return float(abs(m[1] - m[0]) / np.sqrt((s[0] ** 2 + s[1] ** 2) / 2))


def speaker_anchors(rows: list[dict]) -> dict:
    """Median F1 per stressed anchor vowel + the anchor QA verdict."""
    role, trig, f1 = _col(rows, 'role', str), _col(rows, 'trigger', str), _col(rows, 'f1_m50')
    out: dict = {}
    per = {}
    for v in ANCHOR_VOWELS:
        s = f1[(role == 'anchor') & (trig == v)]
        s = s[np.isfinite(s)]
        per[v] = s
        ok = len(s) >= MIN_ANCHOR_N
        out[f'F1_{v}'] = float(np.median(s)) if ok else np.nan
        out[f'n_{v}'] = int(len(s))
    out['gap_e_eps'] = out['F1_ɛ'] - out['F1_e']
    out['gap_o_open'] = out['F1_ɔ'] - out['F1_o']
    out['gap_e_a'] = out['F1_a'] - out['F1_e']
    out['d_e_eps'] = _cohen_d(per['e'], per['ɛ'])
    out['d_o_open'] = _cohen_d(per['o'], per['ɔ'])
    out['anchor_ok'] = int(
        np.isfinite(out['gap_e_eps']) and out['gap_e_eps'] >= MIN_GAP_HZ
        and (not np.isfinite(out['d_e_eps']) or out['d_e_eps'] >= MIN_D)
        and np.isfinite(out['gap_e_a']) and out['gap_e_a'] >= MIN_GAP_AE)
    return out


def normalize_tokens(rows: list[dict], anchors: dict | None = None) -> list[dict]:
    """Add Lobanov z1/z2, openness h, raising g and the trigger's z1 to each row."""
    if not rows:
        return rows
    anc = anchors or speaker_anchors(rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        f1, f2 = _col(rows, 'f1_m50'), _col(rows, 'f2_m50')
        mu, sd = np.nanmean(f1), np.nanstd(f1)
        sd = sd if sd > 0 else 1.0
        mu2, sd2 = np.nanmean(f2), np.nanstd(f2) or 1.0
        trig_z = {v: (anc[f'F1_{v}'] - mu) / sd for v in ANCHOR_VOWELS}
        out = []
        for r, a, b in zip(rows, f1, f2):
            o = r['letter'] == 'o'
            closed = anc['F1_o'] if o else anc['F1_e']
            openv = anc['F1_ɔ'] if o else anc['F1_ɛ']
            high = anc['F1_u'] if o else anc['F1_i']
            rng, rng2 = openv - closed, high - closed
            h = (a - closed) / rng if np.isfinite(rng) and abs(rng) > 1e-6 else np.nan
            g = (a - closed) / rng2 if np.isfinite(rng2) and abs(rng2) > 1e-6 else np.nan
            out.append({**r, 'z1': (a - mu) / sd, 'z2': (b - mu2) / sd2,
                        'h': float(np.clip(h, *H_CLIP)) if np.isfinite(h) else np.nan,
                        'g': float(np.clip(g, -1.0, 2.0)) if np.isfinite(g) else np.nan,
                        'z1_trigger': trig_z.get(r.get('stressed_ipa'), np.nan)})
    return out


def speaker_descriptors(rows: list[dict], anchors: dict | None = None,
                        normalized: bool = False) -> dict:
    """Per-speaker harmony descriptors (OI, HI, open rates, slope, gates...)."""
    anc = anchors or speaker_anchors(rows)
    toks = rows if normalized else normalize_tokens(rows, anc)
    d: dict = {'n_tokens': len(toks), **{k: anc[k] for k in
               ('anchor_ok', 'gap_e_eps', 'gap_o_open', 'd_e_eps')}}
    pre = [t for t in toks if t['role'] == 'pretonic' and 'esx' not in t['flags']]
    trig = _col(pre, 'trigger', str)
    letter = _col(pre, 'letter', str)
    sipa = _col(pre, 'stressed_ipa', str)
    h = _col(pre, 'h')

    def stats(mask):
        x = h[mask]
        x = x[np.isfinite(x)]
        return (len(x), float(np.mean(x)) if len(x) else np.nan,
                float(np.median(x)) if len(x) else np.nan,
                float(np.mean(x > 0.5)) if len(x) else np.nan, x)

    for ctx in ('T_low', 'T_mid', 'T_high'):
        for vw, m in (('both', trig == ctx), ('e', (trig == ctx) & (letter == 'e')),
                      ('o', (trig == ctx) & (letter == 'o'))):
            n, mean, med, orate, x = stats(m)
            k = f'{ctx}_{vw}'
            d[f'n_{k}'], d[f'h_mean_{k}'], d[f'h_med_{k}'], d[f'open_rate_{k}'] = n, mean, med, orate
            if vw == 'both':
                d[f'ashman_{ctx}'] = ashman_d(x)
    for trg, name in ((('ɛ', 'ɔ'), 'T_lowmid'), (('a',), 'T_lowa')):
        n, mean, _, orate, _ = stats(np.isin(sipa, trg))
        d[f'n_{name}'], d[f'h_mean_{name}'], d[f'open_rate_{name}'] = n, mean, orate

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        d['OI'] = d['h_mean_T_low_both']
        d['HI'] = d['OI'] - d['h_mean_T_mid_both']
        d['HI_lowmid'] = d['h_mean_T_lowmid'] - d['h_mean_T_mid_both']
        d['open_rate_low'] = d['open_rate_T_low_both']
        d['open_rate_mid'] = d['open_rate_T_mid_both']
        d['OR_delta'] = d['open_rate_low'] - d['open_rate_mid']
        d['ashman_low'] = d['ashman_T_low']
        # raising uses every pretonic token, <esx> included (as in Exp 56)
        pre_all = [t for t in toks if t['role'] == 'pretonic']
        trig_all, g_all = _col(pre_all, 'trigger', str), _col(pre_all, 'g')
        d['g_mean_T_high'] = float(np.nanmean(g_all[trig_all == 'T_high'])) if (trig_all == 'T_high').any() else np.nan
        d['g_mean_T_mid'] = float(np.nanmean(g_all[trig_all == 'T_mid'])) if (trig_all == 'T_mid').any() else np.nan
        d['RI'] = d['g_mean_T_high'] - d['g_mean_T_mid']

    # gradient-vs-categorical: does h track the trigger's own height?
    lo = trig == 'T_low'
    x, y = _col(pre, 'z1_trigger')[lo], h[lo]
    m = np.isfinite(x) & np.isfinite(y)
    with warnings.catch_warnings():              # near-constant x -> RankWarning
        warnings.simplefilter('ignore')
        d['grad_slope'] = (float(np.polyfit(x[m], y[m], 1)[0])
                           if lo.sum() >= 15 and np.isfinite(x).sum() >= 15 and m.sum() >= 10
                           else np.nan)

    d['gate_ok'] = int(d['anchor_ok'] == 1 and d['n_T_low_both'] >= MIN_LOW
                       and d['n_T_mid_both'] >= MIN_MID)
    return d


def descriptor_vector(desc: dict) -> np.ndarray:
    """Descriptors -> (FEAT_DIM,) vector in FEATURE_NAMES order (NaN if missing)."""
    return np.array([desc.get(k, np.nan) for k in FEATURE_NAMES], dtype=np.float32)


def filter_rare_words(tokens_by_speaker: dict[str, list[dict]],
                      min_word_freq: int = 3) -> dict[str, list[dict]]:
    """Drop tokens whose word type is rarer than `min_word_freq` in the cohort.

    For ASR-transcribed audio these are phonotactically legal hallucinations
    ("farmeanr") that G2P accepts and assigns a trigger that was never said.
    In Exp 56 this removed ~13% of tokens and STRENGTHENED the regional
    contrast (d +0.30 -> +0.35).  Only meaningful over a cohort.
    """
    if min_word_freq <= 1:
        return tokens_by_speaker
    wf = Counter(t['word'] for rows in tokens_by_speaker.values() for t in rows)
    return {s: [t for t in rows if wf[t['word']] >= min_word_freq]
            for s, rows in tokens_by_speaker.items()}


# ── high-level API ───────────────────────────────────────────────────────────

class HarmonyMeasurer:
    """Holds the XLSR model and the G2P lexicon cache across calls."""

    def __init__(self, device: str | None = None, lexicon_path: str | Path | None = None,
                 normalize: bool = True):
        from ..alignment.ctc_char import load_model
        from ..alignment.g2p_stress import Lexicon
        self.model, self.proc, self.char2id, self.device = load_model(device)
        self.lex = Lexicon(str(lexicon_path) if lexicon_path else None)
        self.normalize = normalize

    def measure_audio(self, audio: np.ndarray, text: str | None = None,
                      utt_id: str = '') -> tuple[list[dict], dict]:
        return measure_utterance(audio, text or '', self.lex, self.model, self.proc,
                                 self.char2id, self.device, utt_id=utt_id)

    def measure_file(self, path, text: str | None = None) -> tuple[list[dict], dict]:
        audio = load_audio_norm(path, normalize=self.normalize)
        return self.measure_audio(audio, text, utt_id=str(path))

    def speaker_tokens(self, audio_paths: list, texts: list | None = None) -> list[dict]:
        """Measure every utterance of one speaker -> pooled token rows."""
        texts = list(texts) if texts is not None else [None] * len(audio_paths)
        if len(texts) != len(audio_paths):
            raise ValueError('texts must align 1:1 with audio_paths')
        rows: list[dict] = []
        for p, t in zip(audio_paths, texts):
            try:
                tk, _ = self.measure_file(p, t)
            except Exception as e:                   # one bad file must not kill a speaker
                warnings.warn(f'vowel_harmony: {p}: {e}')
                continue
            rows += tk
        if self.lex.path:
            self.lex.save()
        return rows

    def speaker_vector(self, audio_paths: list, texts: list | None = None
                       ) -> tuple[np.ndarray, dict]:
        """-> ((FEAT_DIM,) vector, {'descriptors', 'n_tokens', 'gate_ok'}).

        Check `meta['gate_ok']`: a speaker failing the anchor or token gates
        has a vector that should not be trusted.
        """
        rows = self.speaker_tokens(audio_paths, texts)
        desc = speaker_descriptors(rows)
        return descriptor_vector(desc), {'descriptors': desc, 'n_tokens': len(rows),
                                         'gate_ok': desc['gate_ok']}

    def cohort(self, speakers: dict[str, list], texts: dict[str, list] | None = None,
               min_word_freq: int = 3) -> dict[str, dict]:
        """Descriptors for many speakers, with the cohort rare-word filter."""
        toks = {s: self.speaker_tokens(p, (texts or {}).get(s))
                for s, p in speakers.items()}
        toks = filter_rare_words(toks, min_word_freq)
        return {s: speaker_descriptors(r) for s, r in toks.items()}


def extract_for_speaker(audio_paths: list, texts: list | None = None,
                        measurer: HarmonyMeasurer | None = None
                        ) -> tuple[np.ndarray, dict]:
    """Functional wrapper around `HarmonyMeasurer.speaker_vector`."""
    hm = measurer or HarmonyMeasurer()
    return hm.speaker_vector(audio_paths, texts)


# ── bundled reference cohort ─────────────────────────────────────────────────

REFERENCE_CSV = Path(__file__).resolve().parents[1] / 'data' / 'vowel_harmony' / 'speakers.csv'


def load_reference_cohort(gate_ok_only: bool = False) -> list[dict]:
    """The Exp 56 per-speaker descriptors (1,153 speakers, 9 corpora).

    Numeric columns come back as floats (NaN where Exp 56 had none); see
    data/vowel_harmony/README.md for the column meanings.  Use it to place a
    new speaker's OI against the NE / SE distributions.
    """
    import csv
    text = {'corpus', 'speaker', 'region', 'macro', 'region_source',
            'transcript_source', 'cluster_gmm'}
    with open(REFERENCE_CSV, newline='', encoding='utf-8') as fh:
        rows = [{k: (v if k in text else float(v) if v != '' else np.nan)
                 for k, v in r.items()} for r in csv.DictReader(fh)]
    if gate_ok_only:
        rows = [r for r in rows if r['gate_ok'] == 1]
    return rows
