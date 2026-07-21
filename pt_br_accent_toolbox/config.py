"""
PT-BR Accent Toolbox — configuration and constants.

All paths can be overridden via environment variables:
    ACCENTS_BASE         → default: /mnt/data/accents
    ZIPA_DIR             → default: {BASE}/zipa_model
    ZIPA_MODEL_FILE      → default: model.onnx
    ZIPA_TOKENS_FILE     → default: tokens.txt
    HF_CACHE_DIR         → default: {BASE}/hf_cache
    ANNOTATIONS_DB       → default: {BASE}/classifier_ui/annotations.db
"""

import os
from pathlib import Path

_BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents'))
BASE = _BASE.resolve()
HF_CACHE = Path(os.environ.get('HF_CACHE_DIR', str(BASE / 'hf_cache')))

# ── audio ────────────────────────────────────────────────────────────────────
SR = 16000
FRAME_MS = 20

# ── ZIPA model (project-specific CTC phone recognizer) ──────────────────────
_ZIPA_DIR = Path(os.environ.get('ZIPA_DIR', str(BASE / 'zipa_model')))
ZIPA_MODEL = _ZIPA_DIR / os.environ.get('ZIPA_MODEL_FILE', 'model.onnx')
ZIPA_TOKENS = _ZIPA_DIR / os.environ.get('ZIPA_TOKENS_FILE', 'tokens.txt')

# ── annotations DB ───────────────────────────────────────────────────────────
ANNOTATIONS_DB = Path(os.environ.get('ANNOTATIONS_DB',
                      str(BASE / 'classifier_ui' / 'annotations.db')))

# ── phone groups for marker tasks ────────────────────────────────────────────
# Candidate phone tokens per marker (IPA strings for both ZIPA and PhoneticXeus)
PHONE_GROUPS = {
    's': ['s', 'z', 'ʃ', 'ʒ'],
    'r': ['ɾ', 'r', 'ʁ', 'ʀ', 'χ', 'x'],
    'dt': ['t', 'd', 't͡ʃ', 'd͡ʒ', 't͡ɕ'],
}

# ── PhoneticXeus ─────────────────────────────────────────────────────────────
PX_MODEL = 'changelinglab/PhoneticXeus'
PX_MAX_S = 25
PX_CHUNK_S = 25
PX_WIN = 2

# ── SSL models ───────────────────────────────────────────────────────────────
SSL_MODELS = {
    'ecapa': 'speechbrain/spkrec-ecapa-voxceleb',
    'xlsr': 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
    'hubert': 'facebook/hubert-large-ls960-ft',
    'w2vbert': 'facebook/w2v-bert-2.0',
}
SSL_SNAP = {
    'xlsr': 'models--jonatasgrosman--wav2vec2-large-xlsr-53-portuguese',
    'hubert': 'models--facebook--hubert-large-ls960-ft',
    'w2vbert': 'models--facebook--w2v-bert-2.0',
}

# ── formant extraction ──────────────────────────────────────────────────────
VOWEL_SET = ['a', 'ɛ', 'e', 'i', 'ɔ', 'o', 'u']
VOWEL_MAP = {
    'a': 'a', 'ɐ': 'a', 'ɑ': 'a', 'ɒ': 'a', 'ʌ': 'a', 'ə': 'a',
    'ɛ': 'ɛ', 'æ': 'ɛ', 'e': 'e',
    'i': 'i', 'ɪ': 'i', 'ɨ': 'i', 'ɯ': 'i',
    'ɔ': 'ɔ', 'o': 'o', 'ø': 'o', 'ɵ': 'o', 'ɤ': 'o',
    'u': 'u', 'ʊ': 'u', 'ʉ': 'u',
}
