---
name: acoustic-features
description: Phonological and acoustic feature extraction with pt_br_accent_toolbox — spectral moments, ZIPA logit probabilities, MFCCs, vowel formants, PhoneticXeus logits, via the FeaturePipeline and CLI. Used when asked to extract acoustic features, compute spectral moments, run ZIPA/PhoneticXeus on clips, compute MFCCs, or extract formants from speech in this repo.
---

# Acoustic / Phonological Feature Extraction

`pt_br_accent_toolbox` extracts per-speaker feature vectors in three families:
**marker-local** (anchored to the S/R/DT phonological markers — see the
`phone-alignment` skill), **vowel** (per-vowel formant structure), and **global**
(whole-utterance embeddings — see the `ssl-embeddings` skill for the SSL ones).

**Working directory**: this checkout lives at `tools/pt-br-accent-toolbox/`
(hyphenated), not `pt_br_accent_toolbox/`. Older docs may reference the old
underscored path — that directory name used to collide with the installed
package name and shadowed it as an empty namespace package (`import
pt_br_accent_toolbox` from `/mnt/data/accents` would silently resolve to
`unknown location` instead of the real code). The hyphenated directory name
can't collide with the Python identifier, so `from pt_br_accent_toolbox
import FeaturePipeline` now works from any cwd, including `/mnt/data/accents`.

## Feature reference

| Name | Dim | Source | Module |
|------|-----|--------|--------|
| `spec` | 6 | FFT power spectrum, 500–8000 Hz band | `features/spectral.py` → `spectral_moments()` |
| `zipa` | 6+N | Spectral moments + windowed ZIPA CTC softmax at marker frames | `features/zipa.py` → `extract_for_markers()` |
| `px` | 13 | PhoneticXeus group-pooled logits at marker frames (s:4, r:6, dt:3) | `features/phoneticxeus.py` → `extract_for_markers()` |
| `formants` | 29 | Per-vowel F1/F2/F3 means (21D) + LTFD (6D) + LTF0 (2D), via parselmouth Burg at ZIPA vowel spikes | `features/formants.py` → `extract_for_speaker()` |
| `mfcc` | 13 | librosa speaker-mean MFCC | `features/mfcc.py` → `speaker_mfcc_mean()` |
| `ssl_*` | 192–2048 | See `ssl-embeddings` skill | `features/ssl.py` |

`spec`'s 6 dims: `[centroid, spread, skewness, kurtosis, peak_freq, log_band_ratio]`
(log ratio of 5–8 kHz to 2–4 kHz energy).

## Recommended path: `FeaturePipeline`

For anything beyond a single ad-hoc clip, use `FeaturePipeline` — it reuses the
ZIPA ONNX session/PhoneticXeus model across calls instead of reloading per-file,
and handles marker collection once per speaker.

```python
from pt_br_accent_toolbox.api.pipeline import FeaturePipeline

pipe = FeaturePipeline()

# One speaker, several utterances
result = pipe.extract_speaker(
    audio_paths=["utt1.wav", "utt2.wav"],
    features=["spec", "formants", "mfcc"],
)
vec, meta = result["formants"]          # (29,) ndarray, {'n_vowels': N}

# Many speakers at once
speakers = {"spk_001": ["a.wav", "b.wav"], "spk_002": ["c.wav"]}
results = pipe.extract(speakers, features=["spec", "px", "ssl_xlsr"])
# results["spec"]["spk_001"] -> (6,) vector
```

Override the phone groups used for `zipa`/`px` marker features with
`groups={'s': [...], 'r': [...], 'dt': [...]}` (defaults to `config.PHONE_GROUPS`).

## CLI

```bash
pt-br-accent-toolbox extract --speakers speakers.json \
    --features spec formants mfcc \
    --out ./results
# speakers.json: {"spk_id": ["path1.wav", "path2.wav"], ...}
# writes ./results/{feature_name}.npz with keys `speaker` and `vector`
```

## Calling individual extractors directly

Useful for a single clip / custom aggregation instead of the full speaker pipeline:

```python
from pt_br_accent_toolbox.features.spectral import spectral_moments
from pt_br_accent_toolbox.features.formants import utterance_vowel_formants, speaker_formant_vector
from pt_br_accent_toolbox.features.mfcc import extract_mfcc, speaker_mfcc_mean

feat = spectral_moments(audio_np_array)   # (6,) float32, audio at any length
```

`features/zipa.py` and `features/phoneticxeus.py` extractors need markers +
model/session objects from the alignment layer — see the `phone-alignment` skill
for how to obtain those, or just go through `FeaturePipeline` which wires it all
together.

## Output convention

`.npz` files saved by the CLI (and by `examples/run_pipeline.py`) use keys
`speaker` (array of speaker IDs) and `vector` (N × D float32 matrix, same row
order). Load with `np.load(path)["vector"]`.

## Gotchas

- Missing/failed markers (e.g. no `zipa`/`px` features because no markers were
  found in the audio) leave that feature key absent from the result dict — check
  `feature_name in result` before indexing, don't assume every requested feature
  is always present.
- `formants` returns `NaN` for vowels never observed in a speaker's audio (missing
  entries in the 21D per-vowel block) — impute before feeding to a classifier (see
  the `classifier-training` skill's `loso_cv(..., impute_strategy=...)`).
