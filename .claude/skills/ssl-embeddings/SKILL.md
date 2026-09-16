---
name: ssl-embeddings
description: Deep SSL embedding extraction with pt_br_accent_toolbox — HuBERT, wav2vec2-Bert, XLS-R, ECAPA-TDNN, via FeaturePipeline or features/ssl.py directly. Used when asked to extract SSL embeddings, run HuBERT/XLS-R/wav2vec-BERT/ECAPA features, or compute speaker-level deep embeddings in this repo.
---

# SSL / Deep Embedding Extraction

**Working directory**: this checkout lives at `tools/pt-br-accent-toolbox/`
(hyphenated) so it can never shadow the `pt_br_accent_toolbox` package name —
`from pt_br_accent_toolbox import ...` now works from any cwd, including
`/mnt/data/accents`.

`pt_br_accent_toolbox.features.ssl` extracts mean-pooled self-supervised embeddings
across a speaker's utterances. Four backbones are pre-wired:

| Name | HuggingFace repo | Output dim |
|------|-------------------|-----------|
| `ecapa` | `speechbrain/spkrec-ecapa-voxceleb` | 192 |
| `xlsr` | `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese` | 1024 |
| `hubert` | `facebook/hubert-large-ls960-ft` | 1024 |
| `w2vbert` | `facebook/w2v-bert-2.0` | 2048 |

Repo IDs live in `config.SSL_MODELS`; snapshot directory names for the
`local_files_only` cache lookup live in `config.SSL_SNAP`.

## Setup — models must be pre-downloaded

Unlike the ZIPA model, `load_model()` here loads with `local_files_only=True` from
`$HF_CACHE_DIR` (default `$ACCENTS_BASE/hf_cache`) — it will **not** silently
auto-download on first use. Run this once per machine:

```bash
python tools/download_models.py ssl                       # all 4
python tools/download_models.py ssl --ssl-models hubert xlsr   # subset
```

## Recommended path: `FeaturePipeline`

```python
from pt_br_accent_toolbox.api.pipeline import FeaturePipeline

pipe = FeaturePipeline()
result = pipe.extract_speaker(
    audio_paths=["utt1.wav", "utt2.wav"],
    features=["ssl_ecapa", "ssl_xlsr"],
)
vec, meta = result["ssl_ecapa"]     # (192,) float32, {'n_utterances': N}
```

CLI: `pt-br-accent-toolbox extract --speakers speakers.json --features ssl_ecapa ssl_hubert --out ./results`

## Calling `features/ssl.py` directly

Useful when reusing one loaded model across many speakers instead of reloading it
per `FeaturePipeline` instance:

```python
from pt_br_accent_toolbox.features.ssl import load_model, speaker_ssl_vector

model, fe = load_model("hubert")          # (model, feature_extractor); fe is None for ecapa
device = "cuda" if torch.cuda.is_available() else "cpu"

for spk, paths in speakers.items():
    vec, n_utt = speaker_ssl_vector(paths, "hubert", model=model, fe=fe, device=device)
```

`speaker_ssl_vector` caps each utterance at 12s (`cap=192000` samples @ 16 kHz by
default) and mean-pools frame-level hidden states (for `hubert`/`xlsr`/`w2vbert`)
or the ECAPA speaker embedding directly. Missing/unreadable audio files are
skipped; if *all* paths fail you get an all-`NaN` vector back, not an exception —
check `n_utterances == 0` before trusting the result.

## Gotchas

- `load_model()` picks CUDA automatically if available (`torch.cuda.is_available()`)
  — pass `device="cpu"` explicitly to force CPU even with a GPU present.
- The CVE-2025-32434 `torch.load` safety-check bypass in `_patch_torch_load()` is
  needed for older `transformers` + `torch < 2.6` combinations that load
  SpeechBrain/HF checkpoints — it's applied automatically on every `load_model()`
  call, no action needed, but don't be surprised to see it if you're reading the
  source.
- `w2vbert` uses `Wav2Vec2BertModel` specifically (not the generic `AutoModel`) —
  if you add a new SSL backbone, check whether it needs a dedicated model class
  the same way.
