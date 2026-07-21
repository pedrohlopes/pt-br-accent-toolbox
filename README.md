# PT-BR Accent Toolbox

Feature extraction and analysis tools for Brazilian Portuguese speech research.

> **What it is:** A reusable Python package for phonological marker detection,
> multi-source feature extraction, and speaker-level classification — built around
> real phonetic phenomena in Brazilian Portuguese (s-coda, r-coda, dt-palatalization).
>
> **What it isn't:** A paper-specific replication toolkit. No PE-vs-SP tasks, no
> anti-spoofing metrics, no hardcoded experiment configs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI / Python API                            │
│  pt-br-accent-toolbox extract / markers / phonemes / annotations   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FeaturePipeline                                │
│  Orchestrates lazy model loading + per-speaker feature routing      │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌──────────────────┐
│ spec │ │  zipa  │ │  px  │ │ form   │ │        ssl       │
│  6D  │ │  6+N D │ │ 13D  │ │  29D   │ │ ecapa  xlsr      │
│      │ │        │ │      │ │        │ │ hubert w2vbert   │
└──────┘ └────────┘ └──────┘ └────────┘ └──────────────────┘
    │         │         │         │            │
    │         │         │         │            │
    │         │         │         ▼            │
    │         │         │    ┌────────────┐    │
    │         │         │    │ parselmouth│    │
    │         │         │    │  (Burg)    │    │
    │         │         │    └────────────┘    │
    │         │         │                      │
    ▼         ▼         ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Alignment Layer                                │
│  ZIPA (CTC ONNX)  │  PhoneticXeus (HuggingFace)                    │
│  ─ Phoneme timeline  ─ Marker frame detection ─ Phone group logits  │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature Reference

| Feature | Dimension | Source | Description |
|---------|-----------|--------|-------------|
| `spec` | 6 | Audio FFT | Spectral centroid, spread, skewness, kurtosis, peak freq, log-band ratio |
| `zipa` | 6+N | ZIPA ONNX | Spectral moments + windowed CTC softmax probs at marker frames |
| `px` | 13 | PhoneticXeus | Group-pooled logits (s: 4, r: 6, dt: 3) at marker positions |
| `formants` | 29 | parselmouth | Per-vowel MF (21D) + LTFD (6D) + LTF0 (2D), vowel spikes from ZIPA |
| `mfcc` | 13 | librosa | Speaker-mean MFCC vector |
| `ssl_ecapa` | 192 | SpeechBrain | ECAPA-TDNN speaker embedding |
| `ssl_xlsr` | 1024 | HuggingFace | XLS-R 53-layer Portuguese Wav2Vec2 |
| `ssl_hubert` | 1024 | HuggingFace | Facebook HuBERT large |
| `ssl_w2vbert` | 2048 | HuggingFace | Wav2Vec2-Bert 2.0 |

## Setup

### 1. Download models

```bash
# Download everything (ZIPA, PhoneticXeus, 4 SSL models)
python tools/download_models.py all

# Or download individually:
python tools/download_models.py zipa              # ZIPA CTC phone recognizer
python tools/download_models.py phoneticxeus      # PhoneticXeus from HuggingFace
python tools/download_models.py ssl               # ECAPA + XLS-R + HuBERT + Wav2Vec2-Bert
```

> The ZIPA model is a custom CTC phone recognizer (~1.2 GB). See
> `tools/download_models.py --help` for copy-from-path and URL options.

### 2. Download datasets (optional)

```bash
python tools/download_datasets.py list            # see what's available
python tools/download_datasets.py brspeech_df     # BRSpeech-DF bonafide
python tools/download_datasets.py gneutral        # GneutralSpeech (requires Kaggle)
python tools/download_datasets.py tagarela        # TAGARELA spotify subset
```

### 3. Installation

```bash
cd /mnt/data/accents/pt_br_accent_toolbox
pip install -e .
```

### CLI Usage

```bash
# Extract features for a set of speakers
pt-br-accent-toolbox extract \
    --speakers speakers.json \
    --features formants mfcc spec \
    --out ./results

# Detect phonological markers in an audio file
pt-br-accent-toolbox markers audio.wav

# Get phoneme sequence
pt-br-accent-toolbox phonemes audio.wav

# Query speaker annotations
pt-br-accent-toolbox annotations --marker s_coda --value chiado
```

### Python API

```python
from pt_br_accent_toolbox.api.pipeline import FeaturePipeline
from pt_br_accent_toolbox.config import PHONE_GROUPS

pipe = FeaturePipeline()

# Single speaker
result = pipe.extract_speaker(
    audio_paths=["file1.wav", "file2.wav"],
    features=["formants", "mfcc", "ssl_ecapa"],
)
formant_vec, n_vowels = result["formants"]  # (29,), count

# Multiple speakers, all features
speakers = {
    "spk_001": ["audio/spk1_1.wav", "audio/spk1_2.wav"],
    "spk_002": ["audio/spk2_1.wav"],
}
results = pipe.extract(speakers, features=["spec", "px", "ssl_xlsr"])
# results["spec"]["spk_001"] -> (6,) vector
```

### Classification

```python
import numpy as np
from pt_br_accent_toolbox.classification.loso import loso_cv, compute_eer
from pt_br_accent_toolbox.classification.ablation import ablation_grid, CLASSIFIERS

# Feature matrices (N speakers x D features), labels, speaker IDs
X = np.load("results/formants.npz")["vector"]  # (N, 29)
y = np.array([0, 1, 0, 1, ...])                # binary labels
spks = np.array(["spk_001", "spk_002", ...])    # speaker IDs

# LOSO CV
from sklearn.ensemble import RandomForestClassifier
res = loso_cv(X, y, spks, RandomForestClassifier(n_estimators=300))
print(f"LOSO accuracy: {res['acc']:.3f}")
print(f"EER: {compute_eer(res['labels'], res['scores']):.1f}%")

# Ablation grid (feature x classifier)
results = ablation_grid(
    feature_sets={"formants": X_f, "mfcc": X_m, "ssl_ecapa": X_s},
    labels=y,
    groups=spks,
)
for r in results:
    print(f"{r['feature']:12s} {r['classifier']:3s} → acc={r['acc']:.3f}")
```

## Marker Phone Groups

The tool detects three phonological markers using these IPA phone sets:

| Marker | Phones | Phenomenon |
|--------|--------|------------|
| `s` | s, z, ʃ, ʒ | Sibilant vs. fricative in coda position |
| `r` | ɾ, r, ʁ, ʀ, χ, x | Tap vs. trill vs. fricative vs. uvular |
| `dt` | t, d, tʃ, dʒ, tɕ | Alveolar vs. affricate (palatalization) |

Override with `groups=...` parameter on any extraction function.

## Annotation Database

The `data/annotations` module loads speaker-level annotations from a SQLite database
(default: `../classifier_ui/annotations.db`). Each speaker has optional labels for
`s_coda`, `r_coda`, and `dt_palat` markers.

```python
from pt_br_accent_toolbox.data.annotations import load_annotations, get_annotated_speakers

all_ann = load_annotations()
# {'Spk1': {'s_coda': 'chiado'}, 'Spk2': {'r_coda': 'caipira', ...}}

s_speakers = get_annotated_speakers(marker="s_coda", value="chiado")
# ['Spk1', 'Spk7', ...]
```

## Module Reference

```
pt_br_accent_toolbox/
├── README.md
├── pyproject.toml            Package metadata, dependencies, entry point
├── .gitignore
├── tools/
│   ├── download_models.py    Download ZIPA, PhoneticXeus, SSL models
│   └── download_datasets.py  Download BRSpeech-DF, Gneutral, Tagarela, etc.
└── pt_br_accent_toolbox/
    ├── config.py             Audio SR, model paths, phone groups, SSL config
    ├── alignment/
    │   ├── zipa.py           ZIPA CTC alignment, marker detection, phoneme extraction
    │   └── phoneticxeus.py   PhoneticXeus model loading, forward logits, pooling
    ├── features/
    │   ├── spectral.py       6D spectral moments (pure NumPy)
    │   ├── zipa.py           Combined spectral + ZIPA logits at markers
    │   ├── phoneticxeus.py   PX group-logit features per marker
    │   ├── formants.py       29D vowel formant features (parselmouth Burg)
    │   ├── mfcc.py           Speaker-mean MFCC
    │   └── ssl.py            SSL embeddings (ECAPA, XLS-R, HuBERT, Wav2VecBert)
    ├── classification/
    │   ├── loso.py           LOSO CV loop, EER computation
    │   └── ablation.py       Feature × classifier grid search
    ├── data/
    │   └── annotations.py    SQLite annotation loader
    ├── api/
    │   └── pipeline.py       High-level FeaturePipeline orchestrator
    └── cli/
        └── main.py           CLI entry point
```

## Requirements

- **Python ≥ 3.10**
- **ZIPA ONNX model** (`model.onnx` + `tokens.txt`) — run `tools/download_models.py zipa`
- **GPU** recommended for SSL models (falls back to CPU)
- First PhoneticXeus use requires network access (downloads from HuggingFace)

### Dependencies

`numpy`, `scipy`, `scikit-learn`, `torch`, `torchaudio`, `transformers`,
`onnxruntime-gpu`, `soundfile`, `librosa`, `lhotse`, `speechbrain`, `parselmouth`

## Configuration

All paths can be set via environment variables, no code changes needed:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACCENTS_BASE` | `/mnt/data/accents` | Root data directory |
| `ZIPA_DIR` | `{BASE}/zipa_model` | ZIPA ONNX model + tokens |
| `ZIPA_MODEL_FILE` | `model.onnx` | ONNX model filename |
| `ZIPA_TOKENS_FILE` | `tokens.txt` | Vocabulary filename |
| `HF_CACHE_DIR` | `{BASE}/hf_cache` | HuggingFace model cache |
| `ANNOTATIONS_DB` | `{BASE}/classifier_ui/annotations.db` | Speaker annotations DB |

Example:
```bash
export ACCENTS_BASE=/path/to/data
export ZIPA_MODEL_FILE=model.int8.onnx   # use quantized variant
python tools/download_models.py all
```
