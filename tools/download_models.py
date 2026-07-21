"""
Download/cache models required by pt_br_accent_toolbox.

Usage:
    python tools/download_models.py all          # download everything
    python tools/download_models.py zipa         # ZIPA ONNX model + vocab
    python tools/download_models.py phoneticxeus # PhoneticXeus from HF
    python tools/download_models.py ssl          # all 4 SSL models

Paths:
    Models are placed relative to ACCENTS_BASE (default: /mnt/data/accents).
    Override via env vars: ZIPA_DIR, HF_CACHE_DIR, ACCENTS_BASE.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents')).resolve()
HF_CACHE = Path(os.environ.get('HF_CACHE_DIR', str(BASE / 'hf_cache')))
ZIPA_DIR = Path(os.environ.get('ZIPA_DIR', str(BASE / 'zipa_model')))


# ── helpers ──────────────────────────────────────────────────────────────────

def _info(msg: str):
    print(f'  ✓ {msg}', flush=True)


def _warn(msg: str):
    print(f'  ⚠ {msg}', flush=True)


def _step(n: int, total: int, label: str):
    print(f'[{n}/{total}] {label} ...', flush=True)


# ── ZIPA ─────────────────────────────────────────────────────────────────────

ZIPA_FILES = {
    'model.onnx': 'Full-precision CTC phone recognizer (1.2 GB)',
    'model.int8.onnx': 'INT8-quantized CTC phone recognizer (296 MB)',
    'tokens.txt': 'Phone token vocabulary',
}

# Public HF repo — download by default. Override with ZIPA_HF_REPO env var.
ZIPA_HF_REPO = os.environ.get('ZIPA_HF_REPO', 'pedrohlopes/zipa-ctc-ptbr')
ZIPA_DOWNLOAD_URL = os.environ.get('ZIPA_DOWNLOAD_URL', None)


def check_zipa() -> dict[str, Path | None]:
    """Check which ZIPA files are already present."""
    present: dict[str, Path | None] = {}
    for fname in ZIPA_FILES:
        p = ZIPA_DIR / fname
        present[fname] = p if p.exists() else None
    return present


def download_zipa(force: bool = False) -> bool:
    """
    Download ZIPA model files.

    Strategy (in order):
      1. Copy from a local source path (ZIPA_SOURCE_DIR env var).
      2. Download from ZIPA_DOWNLOAD_URL env var.
      3. Copy from legacy path (BASE / 'zipa_model' already has them).
    """
    present = check_zipa()
    all_present = all(v is not None for v in present.values())

    if all_present and not force:
        _info('ZIPA model already present')
        for fname, desc in ZIPA_FILES.items():
            size = ZIPA_DIR.joinpath(fname).stat().st_size
            _info(f'  {fname}  ({size / 1e6:.0f} MB)')
        return True

    ZIPA_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy 1: copy from local source
    src = os.environ.get('ZIPA_SOURCE_DIR', None)
    if src:
        src_p = Path(src)
        ok = True
        for fname in ZIPA_FILES:
            sf = src_p / fname
            if sf.exists():
                shutil.copy2(sf, ZIPA_DIR / fname)
                _info(f'{fname} copied from {sf}')
            else:
                _warn(f'{fname} not found at {sf}')
                ok = False
        if ok:
            return True

    # Strategy 2: download from URL (raw URL, not HF)
    if ZIPA_DOWNLOAD_URL:
        import urllib.request
        for fname in ZIPA_FILES:
            if present[fname] and not force:
                continue
            url = f'{ZIPA_DOWNLOAD_URL}/{fname}'
            dest = ZIPA_DIR / fname
            _info(f'downloading {url} ...')
            try:
                urllib.request.urlretrieve(url, dest)
                _info(f'{fname} downloaded ({dest.stat().st_size / 1e6:.0f} MB)')
            except Exception as e:
                _warn(f'download failed: {e}')
                return False
        return True

    # Strategy 3: download from HuggingFace Hub (default)
    _info(f'downloading ZIPA model from {ZIPA_HF_REPO} ...')
    try:
        from huggingface_hub import hf_hub_download
        for fname in ZIPA_FILES:
            if present[fname] and not force:
                continue
            local = hf_hub_download(ZIPA_HF_REPO, fname, repo_type='model',
                                    local_dir=ZIPA_DIR, local_dir_use_symlinks=False)
            _info(f'{fname} → {local}')
        _info('ZIPA model downloaded from HuggingFace Hub')
        return True
    except Exception as e:
        _warn(f'HF Hub download failed: {e}')
        print()
        print(f'  Could not download from {ZIPA_HF_REPO}. Options:')
        print(f'    Option A — Set a different HF repo:')
        print(f'      export ZIPA_HF_REPO=your-org/zipa-model')
        print(f'      python tools/download_models.py zipa')
        print()
        print(f'    Option B — Download from a raw URL:')
        print(f'      export ZIPA_DOWNLOAD_URL=https://your-host/models/zipa')
        print(f'      python tools/download_models.py zipa')
        print()
        print(f'    Option C — Copy files manually to: {ZIPA_DIR}/')
        print(f'      Needs: model.onnx (or model.int8.onnx) + tokens.txt')
        print()
        return False


# ── PhoneticXeus ─────────────────────────────────────────────────────────────

PX_REPO = 'changelinglab/PhoneticXeus'


def download_phoneticxeus(force: bool = False) -> bool:
    """Pre-cache PhoneticXeus model from HuggingFace Hub."""
    cache_dir = HF_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = f'models--{PX_REPO.replace("/", "--")}'
    cached = list(cache_dir.glob(f'{cache_key}/snapshots/*/'))
    if cached and not force:
        _info(f'PhoneticXeus already cached ({len(cached)} snapshots)')
        return True

    _info(f'downloading {PX_REPO} from HuggingFace ...')
    try:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(PX_REPO, cache_dir=str(cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(PX_REPO, cache_dir=str(cache_dir))
        _info(f'PhoneticXeus downloaded to {cache_dir / cache_key}')
        return True
    except Exception as e:
        _warn(f'PhoneticXeus download failed: {e}')
        return False


# ── SSL models ───────────────────────────────────────────────────────────────

SSL_REPOS = {
    'ecapa': ('speechbrain/spkrec-ecapa-voxceleb', 'SpeechBrain ECAPA-TDNN'),
    'xlsr': ('jonatasgrosman/wav2vec2-large-xlsr-53-portuguese', 'XLS-R 53 PT'),
    'hubert': ('facebook/hubert-large-ls960-ft', 'HuBERT large LS960'),
    'w2vbert': ('facebook/w2v-bert-2.0', 'Wav2Vec2-Bert 2.0'),
}


def download_ssl(models: list[str] | None = None, force: bool = False) -> bool:
    """Pre-cache SSL models from HuggingFace."""
    targets = list(SSL_REPOS.keys()) if models is None else models
    cache_dir = HF_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for i, name in enumerate(targets):
        if name not in SSL_REPOS:
            _warn(f'unknown SSL model "{name}", skipping')
            ok = False
            continue

        repo, desc = SSL_REPOS[name]
        _step(i + 1, len(targets), f'{desc} ({name})')

        if name == 'ecapa':
            ecapa_dir = cache_dir / 'spkrec-ecapa-voxceleb'
            already = len(list(ecapa_dir.glob('*.ckpt'))) >= 4 and not force
            if already:
                _info(f'ECAPA already cached ({len(list(ecapa_dir.glob("*.ckpt")))} files)')
                continue
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                EncoderClassifier.from_hparams(
                    source=repo,
                    savedir=str(ecapa_dir),
                    run_opts={'device': 'cpu'},
                )
                _info(f'ECAPA downloaded to {ecapa_dir}')
            except Exception as e:
                _warn(f'ECAPA download failed: {e}')
                ok = False
        else:
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo, cache_dir=str(cache_dir))
                _info(f'{repo} cached')
            except Exception as e:
                _warn(f'{repo} download failed: {e}')
                ok = False

    return ok


# ── all ──────────────────────────────────────────────────────────────────────

def download_all(force: bool = False) -> bool:
    """Download everything needed."""
    print('=== ZIPA ONNX model ===')
    z_ok = download_zipa(force)
    print()

    print('=== PhoneticXeus ===')
    px_ok = download_phoneticxeus(force)
    print()

    print('=== SSL models (ECAPA, XLS-R, HuBERT, Wav2Vec2-Bert) ===')
    ssl_ok = download_ssl(force=force)
    print()

    all_ok = z_ok and px_ok and ssl_ok
    print(f'Models: {"✓ all ready" if all_ok else "⚠ some failed"}')
    return all_ok


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download / cache models for pt_br_accent_toolbox',
    )
    parser.add_argument('target', nargs='?', default='all',
                        choices=['all', 'zipa', 'phoneticxeus', 'ssl'],
                        help='What to download (default: all)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Re-download even if cached')
    parser.add_argument('--ssl-models', nargs='+',
                        choices=list(SSL_REPOS.keys()),
                        help='Specific SSL models to download')
    args = parser.parse_args()

    if args.target == 'all':
        ok = download_all(args.force)
    elif args.target == 'zipa':
        ok = download_zipa(args.force)
    elif args.target == 'phoneticxeus':
        ok = download_phoneticxeus(args.force)
    elif args.target == 'ssl':
        ok = download_ssl(args.ssl_models, args.force)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
