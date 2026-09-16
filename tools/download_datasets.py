"""
Download datasets used with pt_br_accent_toolbox.

Usage:
    python tools/download_datasets.py list                # show available datasets
    python tools/download_datasets.py alcaim              # Alcaim / CETUC read speech
    python tools/download_datasets.py coraa               # CORAA (NURC-RE, C-ORAL, TED)
    python tools/download_datasets.py common_voice        # Common Voice pt
    python tools/download_datasets.py all                 # every auto-downloadable set

Each dataset has its own dedicated download script bundled in this same tools/
directory for fine-grained control (per-speaker limits, sentence subsets, splits) —
this tool just wraps them with sensible defaults. Anything after the dataset name is
forwarded verbatim to the underlying script:

    python tools/download_datasets.py alcaim --sentences balanced50 --max-speakers 20

Downloads land in $ACCENTS_BASE (default /mnt/data/accents) unless the wrapped
script is given --out.
"""

import argparse
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent  # pt_br_accent_toolbox/tools/


def _script(name: str) -> Path:
    """Resolve a download script path (bundled in tools/ alongside this file)."""
    return TOOLS_DIR / name


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
    'alcaim': {
        'desc': 'Alcaim / CETUC — 145 h read speech, 100 speakers x 1000 sentences',
        'script': 'download_alcaim.py',
        'args': [],
        'notes': ('Streams the 12.8 GB tarball and filters on the fly. '
                  '--sentences balanced50 pulls the phonetically balanced '
                  '50-sentence subset; --corpus also serves lapsbm / sid / voxforge. '
                  'Transcriptions come from the THLS 1000-sentence list.'),
        'large': True,
    },
    'brspeech_df': {
        'desc': 'BRSpeech-DF bonafide samples (HF: AKCIT-Deepfake/BRSpeech-DF)',
        'script': 'download_brspeech_df.py',
        'args': ['--min_wavs', '5'],
        'notes': 'Downloads FLAC files organized by speaker. ~800 speakers.',
    },
    'certas_palavras': {
        'desc': 'Certas Palavras — isolated word reading (HF: nilc-nlp/certas_palavras)',
        'script': 'download_certas_palavras.py',
        'args': [],
        'notes': 'Downloads via datasets library. ~70 speakers, studio quality.',
    },
    'cml_tts': {
        'desc': 'CML-TTS Portuguese — LibriVox-derived read speech (OpenSLR 146)',
        'script': 'download_cml_tts.py',
        'args': [],
        'notes': 'Streams the 9.7 GB tar.bz; flattens to cml_tts/{speaker}/*.wav.',
        'large': True,
    },
    'colingpb': {
        'desc': 'CoLingPB interview WAVs + pyannote diarization',
        'script': 'download_colingpb.py',
        'args': [],
        'notes': 'Downloads from repositorio.ufpb.br. Requires pyannote + HF token.',
    },
    'common_voice': {
        'desc': 'Common Voice pt — Mozilla crowdsourced read speech',
        'script': 'download_common_voice.py',
        'args': [],
        'notes': ('Mozilla moved distribution to the Mozilla Data Collective in '
                  'Oct 2025, so this pulls from a HF mirror by default; '
                  '--source local organizes a cv_pt.tar.gz you downloaded yourself.'),
        'large': True,
    },
    'coraa': {
        'desc': 'CORAA ASR v1.1 — NURC-RE + C-ORAL Brasil + TEDx sub-corpora',
        'script': 'download_coraa.py',
        'args': ['--split', 'dev'],
        'notes': ('Splits into coraa_nurc / coraa_coral / coraa_ted by speaker. '
                  'Pass --split train for the full 59 GB archive.'),
        'large': True,
    },
    'gneutral': {
        'desc': 'GneutralSpeech male + female (Kaggle: mediatechlab)',
        'script': 'download_gneutral.py',
        'args': [],
        'notes': 'Requires kaggle API credentials (~/.kaggle/kaggle.json).',
    },
    'mlaad': {
        'desc': 'MLAAD pt — multi-TTS synthetic speech (HF: mueller91/MLAAD)',
        'script': 'download_mlaad.py',
        'args': [],
        'notes': ('16 TTS systems under fake/pt/. CC-BY-NC 4.0, auto-gated repo — '
                  'set HF_TOKEN or run `huggingface-cli login` first.'),
    },
    'nurcsp': {
        'desc': 'NURC-SP audio corpus (HF: nilc-nlp/CORAA-NURC-SP-Audio-Corpus)',
        'script': 'download_nurcsp.py',
        'args': [],
        'notes': 'Defaults to dev+test; --splits train adds the ~40 GB train tarball.',
        'large': True,
    },
    'sotaque_brasileiro': {
        'desc': 'Sotaque Brasileiro — crowdsourced PT-BR accent corpus with geo metadata',
        'script': 'download_sotaque_brasileiro.py',
        'args': [],
        'notes': ('Newest GitHub release snapshot (~1.6 GB). --by-state builds the '
                  'per-state tree the accent classifiers expect.'),
    },
    'tagarela': {
        'desc': 'TAGARELA — segmented Brazilian podcast speech (HF: freds0/TAGARELA)',
        'script': 'download_tagarela.py',
        'args': ['--max-shards', '4'],
        'notes': ('1764 parquet shards keyed by Spotify episode ID. Pass '
                  '--episodes @ids.txt to reproduce a curated subset, or raise '
                  '--max-shards for a larger sample.'),
        'large': True,
    },
    'yodas': {
        'desc': 'YODAS pt-BR sample (HF: AdoCleanCode/portuguese_yodas_mfa_aligned)',
        'script': 'download_yodas.py',
        'args': [],
        'notes': 'Small YouTube-derived sample, ~963 utterances, no speaker labels.',
    },
}

# Datasets without a dedicated download script — no public programmatic source.
MANUAL_DATASETS = {
    'fakebraccent': {
        'desc': 'Fake_BrAccent — voice-converted PT-BR speech, 5 accent classes',
        'notes': ('Not publicly distributed. Contact the corpus authors; the '
                  'accents project keeps it under data/synthetic/fake_braccent.'),
    },
    'braccent': {
        'desc': 'BRAccent — PT-BR regional accent corpus (UFPA)',
        'url': 'https://github.com/falabrasil/speech-datasets',
        'notes': ('Distributed on request by the FalaBrasil group at UFPA. '
                  'Layout once obtained: BRAccent/{Mono,Stereo}/{region}/{gender}/.'),
    },
    'nurc_rj': {
        'desc': 'NURC-RJ — spoken PB corpus (Rio de Janeiro)',
        'url': 'https://nurc.letras.ufrj.br/',
        'notes': ('Contact the NURC project. For NURC-SP use the `nurcsp` entry '
                  'above, which is on HuggingFace.'),
    },
    'ynoguti': {
        'desc': 'Ynoguti PB speech corpus (Inatel)',
        'notes': 'Contact the corpus authors; distributed as novabase.zip.',
    },
    'lasas': {
        'desc': 'LASAS — PB speech corpus',
        'notes': 'Contact corpus authors.',
    },
    'blizzard2027': {
        'desc': 'Blizzard-2027 TTS corpora (falar_tts, nurc_tts)',
        'notes': 'Challenge-restricted; obtained through the Blizzard organisers.',
    },
}


def list_datasets():
    """Print available datasets and their status."""
    print('Datasets with auto-download:')
    print()
    for name, ds in sorted(DATASETS.items()):
        script = _script(ds['script'])
        has_script = '✓' if script.exists() else '✗ (script not found)'
        size_tag = '  [bulk — excluded from `all`]' if ds.get('large') else ''
        print(f'  {name}{size_tag}')
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


def download_all(extra_args: list[str] | None = None,
                 include_large: bool = False) -> bool:
    """
    Download the auto-downloadable datasets.

    Corpora marked 'large' (tens of GB each) are skipped unless include_large is
    set, so a bare `all` stays something you can run without planning disk space.
    """
    skipped = [n for n, ds in DATASETS.items() if ds.get('large') and not include_large]
    all_ok = True
    for name in DATASETS:
        if name in skipped:
            continue
        print(f'=== {name} ===')
        ok = download_dataset(name, extra_args)
        print()
        all_ok = all_ok and ok
    if skipped:
        _info(f'skipped bulk corpora: {", ".join(skipped)} '
              f'(add --include-large, or download them individually)')
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
    parser.add_argument('--include-large', action='store_true',
                        help='with "all", also fetch the multi-GB corpora')
    parser.add_argument('extra', nargs=argparse.REMAINDER,
                        help='Extra args passed to the underlying download script')
    args = parser.parse_args()

    if args.target == 'list':
        list_datasets()
        ok = True
    elif args.target == 'all':
        ok = download_all(args.extra, include_large=args.include_large)
    else:
        ok = download_dataset(args.target, args.extra)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
