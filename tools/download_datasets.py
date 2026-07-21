"""
Download datasets used with pt_br_accent_toolbox.

Usage:
    python tools/download_datasets.py list                         # show available datasets
    python tools/download_datasets.py brspeech_df                  # BRSpeech-DF (bonafide)
    python tools/download_datasets.py gneutral                     # GneutralSpeech (male+female)
    python tools/download_datasets.py tagarela                     # TAGARELA spotify subset
    python tools/download_datasets.py colingpb                     # CoLingPB interviews

    python tools/download_datasets.py all                          # download everything

Each dataset has its own dedicated download script at the repo root
for fine-grained control — this tool wraps them with sensible defaults.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents')).resolve()
REPO_ROOT = Path(__file__).resolve().parent.parent  # pt_br_accent_toolbox/


def _script(name: str) -> Path:
    """Resolve a download script path, checking repo root and accents root."""
    candidates = [
        REPO_ROOT.parent / name,       # /mnt/data/accents/{name}
        Path('/mnt/data/accents') / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # return the expected path even if missing


def _info(msg: str):
    print(f'  ✓ {msg}', flush=True)


def _warn(msg: str):
    print(f'  ⚠ {msg}', flush=True)


def _run_script(script_name: str, extra_args: list[str] | None = None):
    """Run an existing download script with optional extra args."""
    script = _script(script_name)
    if not script.exists():
        _warn(f'{script_name} not found at {script}')
        return False

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f'  running: {" ".join(cmd)}', flush=True)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        _info(f'{script_name} completed')
    else:
        _warn(f'{script_name} failed (exit {result.returncode})')
    return result.returncode == 0


# ── dataset registry ─────────────────────────────────────────────────────────

DATASETS = {
    'brspeech_df': {
        'desc': 'BRSpeech-DF bonafide samples (HF: AKCIT-Deepfake/BRSpeech-DF)',
        'script': 'download_brspeech_df.py',
        'args': ['--min_wavs', '5'],
        'notes': 'Downloads FLAC files organized by speaker. ~800 speakers.',
    },
    'gneutral': {
        'desc': 'GneutralSpeech male + female (Kaggle: mediatechlab)',
        'script': 'download_gneutral.py',
        'args': [],
        'notes': 'Requires kaggle API credentials (~/.kaggle/kaggle.json).',
    },
    'tagarela': {
        'desc': 'TAGARELA spotify-subset episodes (HF: freds0/TAGARELA)',
        'script': 'download_tagarela.py',
        'args': [],
        'notes': 'Scans 1764 parquet shards to find target episodes. Needs fsspec, pyarrow.',
    },
    'colingpb': {
        'desc': 'CoLingPB interview WAVs + pyannote diarization',
        'script': 'download_colingpb.py',
        'args': [],
        'notes': 'Downloads from repositorio.ufpb.br. Requires pyannote + HF token.',
    },
    'certas_palavras': {
        'desc': 'Certas Palavras — isolated word reading (HF: nilc-nlp/certas_palavras)',
        'script': 'download_certas_palavras.py',
        'args': [],
        'notes': 'Downloads via datasets library. ~70 speakers, studio quality.',
    },
}

# Datasets without a dedicated download script (manual setup for now)
MANUAL_DATASETS = {
    'mlaad': {
        'desc': 'MLAAD PT-BR — multi-TTS fake audio (HF: mozilla-foundation/mlaad)',
        'url': 'https://huggingface.co/datasets/mozilla-foundation/mlaad',
        'notes': 'Filter by lang=pt. 16 TTS systems. Use datasets library to download.',
    },
    'fakebraccent': {
        'desc': 'FakeBrAccent — voice-converted PT-BR speech (Udinese corpus)',
        'notes': 'Contact corpus authors for access. See accent_detection_fakebr/.',
    },
    'alcaim': {
        'desc': 'ALCaim — PB accent corpus (Celle et al.)',
        'url': 'https://www.sketchengine.eu/alcaim-portuguese-corpus/',
        'notes': 'Contact corpus maintainers for access.',
    },
    'nurc': {
        'desc': 'NURC-RJ / NURC-SP — spoken PB corpus',
        'url': 'https://nurc.letras.ufrj.br/',
        'notes': 'Contact NURC project for access.',
    },
    'lasas': {
        'desc': 'LASAS — PB speech corpus',
        'notes': 'Contact corpus authors.',
    },
    'braccent': {
        'desc': 'BRAccent — PT-BR accent corpus',
        'notes': 'Available via UFPA. Check BRAccent/ directory.',
    },
    'common_voice': {
        'desc': 'Common Voice pt — Mozilla crowdsourced corpus',
        'url': 'https://commonvoice.mozilla.org/',
        'notes': 'Download cv_pt.tar.gz, then run extract_cv.py to organize.',
    },
    'cml_tts': {
        'desc': 'CML TTS PT-BR — synthetic speech',
        'notes': 'Download cml_tts_pt.tar.bz, then run extract_cml_tts.py.',
    },
    'coraal': {
        'desc': 'CORAAL — Portuguese speech collection (NURC+CORAL+TED)',
        'notes': 'Obtain coraa_train.zip, then run extract_coraa.py.',
    },
    'sotaque_brasileiro': {
        'desc': 'Sotaque Brasileiro — PT-BR accent dataset',
        'notes': 'Download sotaque_brasileiro.zip and extract.',
    },
}


def list_datasets():
    """Print available datasets and their status."""
    print('Datasets with auto-download:')
    print()
    for name, ds in sorted(DATASETS.items()):
        script = _script(ds['script'])
        has_script = '✓' if script.exists() else '✗ (script not found)'
        print(f'  {name}')
        print(f'    {ds["desc"]}')
        print(f'    script: {ds["script"]} {has_script}')
        if ds['args']:
            print(f'    args:   {" ".join(ds["args"])}')
        print(f'    notes:  {ds["notes"]}')
        print()

    print('Datasets requiring manual setup:')
    print()
    for name, ds in sorted(MANUAL_DATASETS.items()):
        print(f'  {name}')
        print(f'    {ds["desc"]}')
        if ds.get('url'):
            print(f'    URL: {ds["url"]}')
        print(f'    notes: {ds["notes"]}')
        print()


def download_dataset(name: str, extra_args: list[str] | None = None) -> bool:
    """Download a specific dataset."""
    if name in DATASETS:
        ds = DATASETS[name]
        cli_args = list(ds.get('args', []))
        if extra_args:
            cli_args.extend(extra_args)
        return _run_script(ds['script'], cli_args)
    elif name in MANUAL_DATASETS:
        ds = MANUAL_DATASETS[name]
        print(f'  {name} requires manual download:')
        print(f'  {ds["desc"]}')
        if ds.get('url'):
            print(f'  URL: {ds["url"]}')
        print(f'  {ds["notes"]}')
        return False
    else:
        _warn(f'unknown dataset "{name}"')
        return False


def download_all(extra_args: list[str] | None = None) -> bool:
    """Download all auto-downloadable datasets."""
    all_ok = True
    for name in DATASETS:
        print(f'=== {name} ===')
        ok = download_dataset(name, extra_args)
        print()
        all_ok = all_ok and ok
    if all_ok:
        _info('all datasets downloaded')
    else:
        _warn('some datasets failed — check output above')
    return all_ok


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for pt_br_accent_toolbox',
    )
    all_keys = sorted(set(DATASETS.keys()) | set(MANUAL_DATASETS.keys()))
    parser.add_argument('target', nargs='?', default='list',
                        choices=['list', 'all'] + all_keys,
                        help='Dataset to download (default: list)')
    parser.add_argument('extra', nargs=argparse.REMAINDER,
                        help='Extra args passed to the underlying download script')
    args = parser.parse_args()

    if args.target == 'list':
        list_datasets()
        ok = True
    elif args.target == 'all':
        ok = download_all(args.extra)
    else:
        ok = download_dataset(args.target, args.extra)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
