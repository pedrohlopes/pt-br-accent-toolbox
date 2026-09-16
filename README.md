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
                         Audio file(s)
                     load_audio() → 16 kHz
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────────┐     ┌─────▼──────┐      ┌─────▼──────┐
    │ ZIPA (ONNX) │     │  librosa   │      │ SSL models │
    │ CTC align   │     │  MFCC      │      │ ECAPA etc. │
    │ markers     │     └────────────┘      └─────┬──────┘
    │ spikes      │                                │
    │ logprobs    │                                │
    └────┬────────┘                                │
         │                                         │
    ┌────▼────────┐                                │
    │PhoneticXeus │                                │
    │ phone logits│                                │
    └────┬────────┘                                │
         │                                         │
    ┌────▼────────────────────┬────────────────────▼──┐
    │              FeaturePipeline                     │
    │  spec  zipa  px   │  formants  │  mfcc  ssl_*  │
    │  ←── marker ──→   │ ← vowel →  │ ←─ global ──→│
    └──────────────────────┬──────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────┐
    │              Classification                      │
    │  loso_cv()  ·  ablation_grid() ·  compute_eer() │
    └─────────────────────────────────────────────────┘
```

Three feature families: **marker-local** (aligned to phonological events), **vowel**
(per-vowel formant structure), and **global** (full-utterance embeddings).

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

## Quick Start (new machine)

```bash
# 1. Clone + install
git clone <repo-url> pt_br_accent_toolbox
cd pt_br_accent_toolbox
pip install -e .

# 2. Download models (auto-downloads from HuggingFace Hub)
python tools/download_models.py all

# 3. Done. Start extracting features:
pt-br-accent-toolbox markers my_audio.wav
```

## Setup

### 1. Install the package

```bash
cd pt_br_accent_toolbox
pip install -e .
```

### 2. Download models

Models are auto-downloaded from HuggingFace Hub. No tokens or accounts needed.

```bash
# Download everything (ZIPA ~1.2 GB, PhoneticXeus, 4 SSL models)
python tools/download_models.py all

# Or individually:
python tools/download_models.py zipa              # ZIPA CTC phone recognizer
python tools/download_models.py phoneticxeus      # PhoneticXeus from HuggingFace
python tools/download_models.py ssl               # ECAPA + XLS-R + HuBERT + Wav2Vec2-Bert
```

The ZIPA model (`pedrohlopes/zipa-ctc-ptbr`) is a custom CTC phone recognizer
exported to ONNX. Use `model.int8.onnx` for a smaller download (296 MB vs 1.2 GB):

```bash
export ZIPA_MODEL_FILE=model.int8.onnx
python tools/download_models.py zipa
```

### 3. Download datasets (optional)

19 corpora are registered: **13 with auto-download scripts** bundled in `tools/`,
6 that are only available on request from their authors.
`python tools/download_datasets.py list` shows both lists with their notes.

```bash
# Install the extra deps these scripts need
pip install -e ".[datasets]"

python tools/download_datasets.py list        # everything, with notes and caveats
python tools/download_datasets.py all         # the small/medium sets
python tools/download_datasets.py all --include-large   # plus the multi-GB corpora
```

| Dataset | Source | Notes |
|---------|--------|-------|
| `alcaim` | [smt.ufrj.br mirror](https://igormq.github.io/datasets/) | Alcaim / CETUC, 145 h, 100 speakers × 1000 sentences. Streams + filters. |
| `brspeech_df` | HF `AKCIT-Deepfake/BRSpeech-DF` | Bonafide samples by speaker |
| `certas_palavras` | HF `nilc-nlp/certas_palavras` | Isolated word reading, ~70 speakers |
| `cml_tts` | [OpenSLR 146](https://www.openslr.org/146/) | CML-TTS Portuguese, 9.7 GB tar.bz |
| `colingpb` | repositorio.ufpb.br | Interviews + pyannote diarization (needs `HF_TOKEN`) |
| `common_voice` | HF mirror (or local `cv_pt.tar.gz`) | Mozilla moved distribution to the Data Collective in Oct 2025 |
| `coraa` | [nilc-nlp/CORAA](https://github.com/nilc-nlp/CORAA) | Splits into NURC-RE / C-ORAL / TEDx by speaker |
| `gneutral` | Kaggle `mediatechlab` | Needs `~/.kaggle/kaggle.json` |
| `mlaad` | HF `mueller91/MLAAD` | pt slice, 16 TTS systems. CC-BY-NC, gated — needs `HF_TOKEN` |
| `nurcsp` | HF `nilc-nlp/CORAA-NURC-SP-Audio-Corpus` | Spontaneous São Paulo speech |
| `sotaque_brasileiro` | GitHub release snapshots | Crowdsourced, with birth/current city+state metadata |
| `tagarela` | HF `freds0/TAGARELA` | Podcast speech, keyed by Spotify episode ID |
| `yodas` | HF `AdoCleanCode/portuguese_yodas_mfa_aligned` | Small YouTube-derived sample |

Manual-only: `fakebraccent`, `braccent`, `nurc_rj`, `ynoguti`, `lasas`, `blizzard2027`.

Every script writes into `$ACCENTS_BASE` (default `/mnt/data/accents`) unless given
`--out`, and anything you pass after the dataset name is forwarded verbatim:

```bash
python tools/download_datasets.py alcaim --sentences balanced50 --max-speakers 20
python tools/download_datasets.py coraa --split train --subsets nurc,coral
python tools/download_datasets.py mlaad --systems "OpenAI TTS-1 HD" --max-per-system 50
```

#### Alcaim / CETUC

The corpus this project leans on hardest gets the most options. The full archive is
12.8 GB, so `download_alcaim.py` **streams** it and writes only the members that
survive the filters — and stops reading as soon as a `--max-speakers` quota is full,
since speakers are stored as contiguous blocks:

```bash
# everything
python tools/download_alcaim.py

# the phonetically balanced 50-sentence subset used across the papers
python tools/download_alcaim.py --sentences balanced50

# 20 speakers, 10 per gender, sentences 1-50
python tools/download_alcaim.py --max-speakers 20 --balance-gender --sentences 1-50

# named speakers, no per-utterance .txt files
python tools/download_alcaim.py --speakers Alcione_F018,Aislam_M001 --no-transcripts

# sibling corpora on the same mirror
python tools/download_alcaim.py --corpus lapsbm      # LapsBM (FalaBrasil/UFPA)
python tools/download_alcaim.py --corpus sid         # Sidney
python tools/download_alcaim.py --corpus voxforge    # VoxForge pt-BR
```

Output:

```
alcaim/
  sentences.txt            the 1000-sentence list, re-encoded latin-1 -> UTF-8
  sentences_subset.txt     the sentences kept by --sentences (+ .ids, 1-based)
  metadata.csv             filename, speaker, gender, sentence_id, text
  {Speaker}_{G}{NNN}/
    {G}{NNN}-{IIII}.wav    16 kHz mono, IIII is the 0-based sentence index
    {G}{NNN}-{IIII}.txt    lowercase, unpunctuated transcription
```

The transcriptions are the THLS 1000-sentence list
([gitlab.com/lfelipesv/1000-sentences-thls-dataset](https://gitlab.com/lfelipesv/1000-sentences-thls-dataset)),
fetched automatically and re-encoded to UTF-8; utterance index `IIII` is the 0-based
line number into it. `--sentences balanced50` reproduces the greedy phonetically
balanced subset the project uses as its TTS prompt set (it selects on the same
grapheme/digraph unit distribution, so it returns exactly the project's
`sentences_50.txt`). Grab just the sentence list with `--sentences-only`.

## Usage

### CLI

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

# Ablation grid (feature x classifier) — CLASSIFIERS has rf, gb, lr, svm, xgb
results = ablation_grid(
    feature_sets={"formants": X_f, "mfcc": X_m, "ssl_ecapa": X_s},
    labels=y,
    groups=spks,
)
for r in results:
    print(f"{r['feature']:12s} {r['classifier']:3s} → acc={r['acc']:.3f}")

# Persist a trained model (+ metadata) and reload it later
from pt_br_accent_toolbox.classification.persistence import save_model, load_model

clf = CLASSIFIERS["xgb"]().fit(X, y)
save_model("models/s_coda", clf, meta={"classes": [0, 1], "feat_names": ["f0", "f1", ...]})
clf2, meta = load_model("models/s_coda")
```

## Marker Phone Groups

The tool detects three phonological markers using these IPA phone sets:

| Marker | Phones | Phenomenon |
|--------|--------|------------|
| `s` | s, z, ʃ, ʒ | Sibilant vs. fricative in coda position |
| `r` | ɾ, r, ʁ, ʀ, χ, x | Tap vs. trill vs. fricative vs. uvular |
| `dt` | t, d, tʃ, dʒ, tɕ | Alveolar vs. affricate (palatalization) |

Override with `groups=...` parameter on any extraction function.

## Speaker Annotations

115 natural PT-BR speakers across nine corpora, hand-labelled for the three
markers. The labels ship with the package as CSVs
(`pt_br_accent_toolbox/data/annotations/`), so a fresh clone reproduces the
labelled cohort with no extra downloads — read that folder's `README.md` for the
label vocabulary and the caveats (selection effects, ties, agreement).

| File | Grain |
|------|-------|
| `annotations.csv` | one row per (dataset, speaker, annotator) — the raw labels |
| `speakers.csv` | one row per (dataset, speaker) — majority label + agreement counts |
| `summary.csv` | counts per dataset × marker × value |
| `todo.csv` | speakers with a marker still unlabelled |

```python
from pt_br_accent_toolbox.data.annotations import (
    load_annotations, load_annotation_rows, get_annotated_speakers)

load_annotations()
# {'Alcione_F018': {'s_coda': 'chiado'}, '5739': {'s_coda': 'sibilant', ...}}

get_annotated_speakers(marker="s_coda", value="chiado")
# ['Alcione_F018', 'Aislam_M001', ...]

load_annotation_rows()          # raw rows: keeps dataset, annotator, source, notes
```

```bash
pt-br-accent-toolbox annotations --marker r_coda --value caipira
pt-br-accent-toolbox annotations --rows --marker s_coda --value chiado
```

`load_annotations()` prefers the SQLite store at `$ANNOTATIONS_DB` when it exists
(the annotation UI writes there) and falls back to the bundled CSV otherwise. It is
keyed by speaker ID alone, which merges the four IDs that appear in two corpora —
use `load_annotation_rows()` when the corpus matters. Re-snapshot the CSVs after
annotating:

```bash
python tools/export_annotations.py --db /path/to/classifier_ui/annotations.db
```

## Module Reference

```
pt_br_accent_toolbox/
├── README.md
├── pyproject.toml            Package metadata, dependencies, entry point
├── .gitignore
├── .claude/skills/           Agent skills documenting this package's own API —
│                             acoustic-features, phone-alignment, ssl-embeddings,
│                             classifier-training (ships with the repo; Claude Code
│                             picks these up automatically on clone)
├── tools/
│   ├── _common.py            Resumable HTTP, streaming tar readers, ACCENTS_BASE paths
│   ├── download_models.py    Download ZIPA, PhoneticXeus, SSL models
│   ├── download_datasets.py  Registry + wrapper over every download_*.py below
│   ├── download_alcaim.py    Alcaim/CETUC (+ LapsBM, Sid, VoxForge) — stream & filter
│   ├── download_brspeech_df.py
│   ├── download_certas_palavras.py
│   ├── download_cml_tts.py
│   ├── download_colingpb.py
│   ├── download_common_voice.py
│   ├── download_coraa.py     -> coraa_nurc / coraa_coral / coraa_ted
│   ├── download_gneutral.py
│   ├── download_mlaad.py
│   ├── download_nurcsp.py
│   ├── download_sotaque_brasileiro.py
│   ├── download_tagarela.py
│   ├── download_yodas.py
│   └── export_annotations.py Snapshot the annotation DB -> data/annotations/*.csv
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
    │   ├── ablation.py       Feature × classifier grid search (rf, gb, lr, svm, xgb)
    │   └── persistence.py    save_model / load_model (pickle + meta.json)
    ├── data/
    │   ├── annotations.py    Annotation loader (SQLite, CSV fallback)
    │   └── annotations/      Bundled label snapshot + its README
    ├── api/
    │   └── pipeline.py       High-level FeaturePipeline orchestrator
    └── cli/
        └── main.py           CLI entry point
```

## Requirements

- **Python ≥ 3.10**
- **Models** — run `python tools/download_models.py all` (auto-downloads from HF Hub)
- **GPU** recommended for SSL models (falls back to CPU)
- Network access required for first-time model downloads

### Dependencies

`numpy`, `scikit-learn`, `torch`, `transformers`, `onnxruntime`,
`soundfile`, `librosa`, `lhotse`, `speechbrain`,
`praat-parselmouth`, `huggingface-hub`

By default the toolbox installs CPU-only `onnxruntime` (works everywhere, including
machines without an NVIDIA GPU). For GPU-accelerated ZIPA inference:

```bash
pip install -e ".[gpu]"
pip uninstall -y onnxruntime   # onnxruntime and onnxruntime-gpu both own the
                                # `onnxruntime` import name; keep only one installed
```

`make_session()` always requests `CUDAExecutionProvider` first and falls back to
`CPUExecutionProvider` automatically if no GPU build is installed — no code changes
needed either way.

### Git Repository

```bash
git remote add origin <your-upstream-url>
git push -u origin main
```

## Configuration

All paths are set via environment variables — no code changes needed.
The package works with defaults on any machine after running the download tools.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACCENTS_BASE` | `/mnt/data/accents` | Root data directory |
| `ZIPA_DIR` | `{BASE}/zipa_model` | ZIPA ONNX model + tokens location |
| `ZIPA_MODEL_FILE` | `model.onnx` | ONNX model filename (`model.int8.onnx` for quantized) |
| `ZIPA_TOKENS_FILE` | `tokens.txt` | Vocabulary filename |
| `ZIPA_HF_REPO` | `pedrohlopes/zipa-ctc-ptbr` | HF Hub repo for ZIPA model download |
| `ZIPA_DOWNLOAD_URL` | — | Raw URL base for ZIPA download (alternative to HF) |
| `ZIPA_SOURCE_DIR` | — | Local directory to copy ZIPA from |
| `HF_CACHE_DIR` | `{BASE}/hf_cache` | HuggingFace model cache directory |
| `ANNOTATIONS_DB` | `{BASE}/classifier_ui/annotations.db` | Speaker annotations SQLite DB |

### Examples

```bash
# Default setup (auto-downloads ZIPA from HF Hub):
python tools/download_models.py all

# Use quantized ZIPA model (296 MB instead of 1.2 GB):
export ZIPA_MODEL_FILE=model.int8.onnx
python tools/download_models.py zipa

# Custom data root:
export ACCENTS_BASE=/path/to/my_data
export HF_CACHE_DIR=/path/to/my_data/hf_cache
python tools/download_models.py all

# Copy ZIPA from another machine:
export ZIPA_SOURCE_DIR=/mnt/old_server/models/zipa
python tools/download_models.py zipa

# Use a different HF repo:
export ZIPA_HF_REPO=my-org/my-zipa-model
python tools/download_models.py zipa
```
