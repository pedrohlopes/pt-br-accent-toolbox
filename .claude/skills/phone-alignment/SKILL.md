---
name: phone-alignment
description: ZIPA CTC forced alignment and phonological marker (S-coda, R-coda, DT-palatalization) detection using the pt_br_accent_toolbox package. Used when asked to align audio, extract phone spikes, detect phonological markers, get a phoneme sequence, or run the ZIPA alignment pipeline in this repo.
---

# Phone Alignment & Marker Detection

**Working directory**: this checkout lives at `tools/pt-br-accent-toolbox/`
(hyphenated) so it can never shadow the `pt_br_accent_toolbox` package name —
`from pt_br_accent_toolbox import ...` now works from any cwd, including
`/mnt/data/accents`.

`pt_br_accent_toolbox.alignment.zipa` runs ZIPA (a CTC phone recognizer exported to
ONNX, 50 Hz frame rate / 20 ms per frame) over 16 kHz Brazilian Portuguese audio and
detects three phonological markers from the resulting phone-spike sequence:

| Marker | Phenomenon | Detection rule |
|--------|-----------|-----------------|
| `s` | Sibilant vs. fricative in coda position | `phone in {s, ʃ}` and next phone is not a vowel |
| `r` | Tap / trill / fricative / uvular in coda | `phone in {ɾ, ʁ, ʀ, ħ, χ, r, h, x}` and next phone is not a vowel |
| `dt` | Alveolar vs. affricate (palatalization) | `phone in {t, d}` and next phone in `{i, ɪ, ʃ, ʒ, j}` |

Phone sets are `PHONE_GROUPS` in `pt_br_accent_toolbox/config.py` and can be
overridden per-call (`groups=...`).

## Setup

Requires the ZIPA model — see the repo README ("Download models"). By default it's
looked up at `$ACCENTS_BASE/zipa_model/model.onnx` (or `$ZIPA_DIR` /
`$ZIPA_MODEL_FILE` directly). Everything here works from any directory; no
repo-specific paths are hardcoded.

```bash
python tools/download_models.py zipa
```

## CLI (fastest path for a single file)

```bash
pt-br-accent-toolbox markers audio.wav              # phonological marker frames → stdout JSON
pt-br-accent-toolbox markers audio.wav --out out.json
pt-br-accent-toolbox phonemes audio.wav              # full CTC-greedy phone sequence
```

`markers` output: `{'n_markers': N, 'markers': [{'marker', 'frame', 'time_s', 'phone', 'next_phone'}, ...]}`.

## Python API

```python
from pt_br_accent_toolbox.alignment.zipa import (
    load_audio, load_vocab, make_session, make_fbank_extractor,
    utterance_logprobs, get_spikes, ctc_greedy_decode,
    extract_marker_frames, extract_phoneme_sequence,
)

audio = load_audio("clip.wav")          # mono float32 @ 16 kHz, resampled if needed

# One-shot marker detection (loads model/vocab internally if not passed)
markers = extract_marker_frames(audio)
# [{'marker': 's', 'frame': 388, 'time_s': 7.76, 'phone': 'ʃ', 'next_phone': 's'}, ...]

# Full phoneme sequence
phones = extract_phoneme_sequence(audio)   # ['a', 'ʃ', 's', ...]
```

### Reusing the session across many files (avoids reloading the ONNX model each call)

```python
sess = make_session()                 # or make_session(model_path=...)
extractor = make_fbank_extractor()
vocab = load_vocab()                  # or load_vocab(tokens_path=...)

for path in audio_paths:
    audio = load_audio(path)
    lp = utterance_logprobs(audio, sess, extractor)     # (T, vocab) log-probs
    spikes = get_spikes(lp, vocab)                       # [(frame, ipa_token), ...]
    markers = extract_marker_frames(audio, sess, extractor)
```

`FeaturePipeline` (see the `acoustic-features` skill) wraps this session-reuse
pattern automatically when extracting features for many speakers at once — prefer
it over calling `extract_marker_frames` per-file in a loop for anything beyond a
single audio file.

## Known pitfalls

1. **Vocab direction**: `load_vocab()` returns `{id: token}` (int → IPA string) —
   this is what `get_spikes`/`ctc_greedy_decode` expect. Some feature-extraction
   code needs the *reverse* map (`token → id`); build it explicitly with
   `{v: k for k, v in vocab.items()}` rather than assuming a shared convention —
   `FeaturePipeline` does this internally already.
2. **Frame rate**: ZIPA runs at 50 Hz. `time_s = frame * FRAME_MS / 1000` (config
   `FRAME_MS = 20`). Don't assume any other hop size.
3. **Audio**: everything must be 16 kHz mono float32 — `load_audio()` handles
   resampling (via soundfile, falling back to librosa) and mono-downmixing for you;
   don't feed it raw multi-channel or non-16k arrays directly to the lower-level
   functions.
4. **CUDA vs CPU**: `make_session()` requests `CUDAExecutionProvider` first and
   silently falls back to `CPUExecutionProvider` if no GPU/CUDA build is present —
   no code changes needed on CPU-only machines, just expect it to be slower.
