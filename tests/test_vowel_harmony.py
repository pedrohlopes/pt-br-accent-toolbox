"""Tests for the vowel-harmony marker (Exp 56 port) — no model downloads needed.

Run with:  python -m pytest tests/test_vowel_harmony.py -v
"""

import shutil

import numpy as np
import pytest

needs_espeak = pytest.mark.skipif(shutil.which('espeak-ng') is None,
                                  reason='espeak-ng not on PATH')


# ── G2P: the trigger comes from the lexicon, never from the audio ────────────

@needs_espeak
def test_g2p_trigger_contexts():
    from pt_br_accent_toolbox.alignment.g2p_stress import Lexicon
    ana = Lexicon(None).analyze(['colega', 'pequeno', 'menino', 'esporte'])

    def pretonic(w):
        (u,) = [u for u in ana[w]['nuclei'] if u['role'] == 'pretonic']
        return u

    assert all(ana[w]['ok'] for w in ana)
    assert (pretonic('colega')['letter'], pretonic('colega')['trigger']) == ('o', 'T_low')
    assert pretonic('pequeno')['trigger'] == 'T_mid'
    assert pretonic('menino')['trigger'] == 'T_high'
    assert 'esx' in pretonic('esporte')['flags']      # es+C onset, excluded from OI


def test_normalize_text():
    from pt_br_accent_toolbox.alignment.g2p_stress import normalize_text
    assert normalize_text('Colega, MOLEQUE!') == 'colega moleque'


# ── character CTC alignment ──────────────────────────────────────────────────

def _peaky_emissions(seq, n_vocab, frames_per=3):
    """log-probs where each target id peaks for one frame, blanks in between."""
    T = len(seq) * frames_per
    lp = np.full((T, n_vocab), -12.0, dtype=np.float32)
    lp[:, 0] = -0.01
    for i, s in enumerate(seq):
        t = i * frames_per + 1
        lp[t, :] = -12.0
        lp[t, s] = -0.01
    return lp


def test_viterbi_numpy_recovers_spikes():
    from pt_br_accent_toolbox.alignment.ctc_char import _viterbi_numpy
    ids = [3, 4, 3, 5]
    labels, scores = _viterbi_numpy(_peaky_emissions(ids, 8), ids)
    assert [l for l in labels if l != 0] == ids
    assert np.argwhere(np.array(labels) == 4).ravel().tolist() == [4]
    assert len(scores) == len(labels)


def test_viterbi_numpy_repeated_char_needs_blank():
    from pt_br_accent_toolbox.alignment.ctc_char import _viterbi_numpy
    ids = [3, 3]                                    # "ss" must be split by a blank
    labels, _ = _viterbi_numpy(_peaky_emissions(ids, 6), ids)
    nz = [i for i, l in enumerate(labels) if l == 3]
    assert len(nz) == 2 and nz[1] - nz[0] > 1


def test_align_spans_and_ctc_text():
    import torch
    from pt_br_accent_toolbox.alignment.ctc_char import align, ctc_text
    char2id = {'<pad>': 0, '|': 1, 'c': 2, 'o': 3, 'l': 4, 'e': 5, 'g': 6, 'a': 7}
    target, dropped = ctc_text('colega!', char2id)
    assert (target, dropped) == ('colega', 1)
    ids = [char2id[c] for c in target]
    spans = align(torch.from_numpy(_peaky_emissions(ids, 8)), target, char2id)
    assert [s[1] for s in spans] == list(target)
    assert all(s[2] <= s[3] < s[4] for s in spans[:-1])   # start <= end < next start


def test_greedy_decode():
    import torch
    from pt_br_accent_toolbox.alignment.ctc_char import greedy_decode
    char2id = {'<pad>': 0, '|': 1, 'o': 2, 'i': 3}
    lp = _peaky_emissions([2, 3, 1, 2, 3], 4)
    assert greedy_decode(torch.from_numpy(lp), char2id) == 'oi oi'


# ── span refinement: must not grow into the stressed vowel ───────────────────

def test_refine_span_respects_neighbour_bound():
    from pt_br_accent_toolbox.features.vowel_harmony import refine_span
    vmass = np.ones(40)                     # a liquid keeps the posterior high
    lo, hi, src = refine_span(10, 11, vmass, lo_bound=8, hi_bound=14)
    assert (lo, hi, src) == (8, 14, 'vocalic')
    lo, hi, _ = refine_span(10, 11, vmass)  # unbounded: capped at MAX_GROW
    assert hi - lo <= 2 * 6 + 1


# ── speaker normalization + descriptors ──────────────────────────────────────

def _synthetic_speaker(open_shift, n_low=30, n_mid=15, seed=0):
    """Anchors e=400 Hz / ɛ=550 Hz (o/ɔ the same); pretonics before an open
    stressed vowel sit `open_shift` of the way from [e] to [ɛ]."""
    rng = np.random.default_rng(seed)
    rows = []

    def tok(role, letter, trigger, f1, sipa=''):
        rows.append({'role': role, 'letter': letter, 'trigger': trigger,
                     'stressed_ipa': sipa, 'flags': '', 'word': 'w',
                     'f1_m50': f1 + rng.normal(0, 10), 'f2_m50': 1500.0})

    for v, f1 in {'a': 750, 'e': 400, 'ɛ': 550, 'o': 400, 'ɔ': 550,
                  'i': 300, 'u': 300}.items():
        for _ in range(12):
            tok('anchor', v if v in 'aeiou' else ('e' if v == 'ɛ' else 'o'), v, f1)
    for _ in range(n_low):
        tok('pretonic', 'e', 'T_low', 400 + open_shift * 150, 'ɛ')
    for _ in range(n_mid):
        tok('pretonic', 'e', 'T_mid', 400 + 0.1 * 150, 'e')
    for _ in range(10):
        tok('pretonic', 'e', 'T_high', 380, 'i')
    return rows


def test_descriptors_separate_lowering_from_not():
    from pt_br_accent_toolbox.features.vowel_harmony import (
        FEAT_DIM, descriptor_vector, speaker_descriptors)
    ne = speaker_descriptors(_synthetic_speaker(0.7))
    se = speaker_descriptors(_synthetic_speaker(0.1))
    assert ne['anchor_ok'] == se['anchor_ok'] == 1
    assert ne['gate_ok'] == se['gate_ok'] == 1
    assert ne['OI'] == pytest.approx(0.7, abs=0.05)
    assert se['HI'] == pytest.approx(0.0, abs=0.05)
    assert ne['HI'] > 0.5 and ne['open_rate_low'] > 0.9
    assert descriptor_vector(ne).shape == (FEAT_DIM,)


def test_gate_fails_without_tokens_or_contrast():
    from pt_br_accent_toolbox.features.vowel_harmony import speaker_descriptors
    assert speaker_descriptors(_synthetic_speaker(0.5, n_low=5))['gate_ok'] == 0
    flat = [dict(r, f1_m50=400.0) for r in _synthetic_speaker(0.5)]
    assert speaker_descriptors(flat)['anchor_ok'] == 0      # [ɛ] not above [e]


def test_filter_rare_words():
    from pt_br_accent_toolbox.features.vowel_harmony import filter_rare_words
    toks = {'a': [{'word': 'colega'}] * 2 + [{'word': 'farmeanr'}],
            'b': [{'word': 'colega'}]}
    out = filter_rare_words(toks, min_word_freq=3)
    assert [t['word'] for t in out['a']] == ['colega', 'colega']
    assert filter_rare_words(toks, 1) is toks


# ── bundled Exp 56 reference cohort ──────────────────────────────────────────

def test_reference_cohort_reproduces_exp56():
    from pt_br_accent_toolbox.features.vowel_harmony import load_reference_cohort
    assert len(load_reference_cohort()) == 1153
    ref = load_reference_cohort(gate_ok_only=True)
    assert len(ref) == 477
    oi = {m: [r['OI'] for r in ref if r['macro'] == m] for m in ('NE', 'SE')}
    assert len(oi['NE']) == 168 and len(oi['SE']) == 112
    assert np.mean(oi['NE']) == pytest.approx(0.646, abs=1e-3)
    assert np.mean(oi['SE']) == pytest.approx(0.357, abs=1e-3)
