"""PT-BR Accent Toolbox — configuration and constants."""

from pathlib import Path

BASE = Path('/mnt/data/accents')
HF_CACHE = BASE / 'hf_cache'

# ── audio ────────────────────────────────────────────────────────────────────
SR = 16000
FRAME_MS = 20

# ── ZIPA model (project-specific CTC phone recognizer) ──────────────────────
ZIPA_MODEL = BASE / 'exp_followups/exp18_long_switch/cache/zipa.onnx'
ZIPA_TOKENS = BASE / 'exp_followups/exp18_long_switch/cache/vocab.txt'

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
