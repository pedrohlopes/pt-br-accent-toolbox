"""
Download the Portuguese slice of MLAAD (Multi-Language Audio Anti-Spoofing Dataset).

    https://huggingface.co/datasets/mueller91/MLAAD

MLAAD is the cross-dataset synthetic-speech pool used in Paper B. The repo is
organized as fake/{lang}/{tts_system}/*.wav, so the PT-BR slice is everything
under fake/pt/. Each system directory also carries a meta.csv.

Output mirrors the upstream layout so paths stay comparable with the published
results:
    mlaad_ptbr/fake/pt/{tts_system}/*.wav

MLAAD is CC-BY-NC 4.0 (non-commercial academic use) and the repo is auto-gated:
you must be logged in to HuggingFace. Export HF_TOKEN or run `huggingface-cli login`.

Usage:
    python tools/download_mlaad.py
    python tools/download_mlaad.py --list-systems
    python tools/download_mlaad.py --systems "OpenAI TTS-1 HD,VoxCPM2"
    python tools/download_mlaad.py --max-per-system 50
    python tools/download_mlaad.py --lang de           # any other language slice

Needs `huggingface_hub`.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import human, info, resolve_out, warn  # noqa: E402

REPO = 'mueller91/MLAAD'


def main():
    ap = argparse.ArgumentParser(description='Download the MLAAD pt slice')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/mlaad_ptbr)')
    ap.add_argument('--lang', default='pt', help='language slice (default: pt)')
    ap.add_argument('--systems', default=None,
                    help='comma-separated TTS system names (default: all)')
    ap.add_argument('--max-per-system', type=int, default=None,
                    help='keep at most N wavs per TTS system')
    ap.add_argument('--list-systems', action='store_true',
                    help='print the TTS systems in this language slice and exit')
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_tree
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

    prefix = f'fake/{args.lang}/'
    print(f'=== MLAAD {args.lang} ===')

    # list_repo_tree scoped to the language directory — listing the whole repo
    # means ~99k paths and several minutes.
    try:
        files = [e.path for e in list_repo_tree(REPO, path_in_repo=f'fake/{args.lang}',
                                                repo_type='dataset', recursive=True,
                                                token=token)
                 if getattr(e, 'size', None) is not None]
    except (GatedRepoError, HfHubHTTPError) as e:
        warn(f'cannot list {REPO}: {e}')
        warn('MLAAD is auto-gated — accept the terms on the dataset page and set '
             'HF_TOKEN (or run `huggingface-cli login`).')
        return 1

    if not files:
        warn(f'no files under {prefix} — check --lang')
        return 1

    systems = sorted({f.split('/')[2] for f in files})
    if args.list_systems:
        for s in systems:
            n = sum(1 for f in files if f.startswith(f'{prefix}{s}/') and f.endswith('.wav'))
            print(f'  {s}\t{n} wavs')
        return 0

    if args.systems:
        wanted = {s.strip() for s in args.systems.split(',') if s.strip()}
        unknown = wanted - set(systems)
        if unknown:
            warn(f'unknown systems: {sorted(unknown)}')
            warn(f'available: {systems}')
            return 1
        files = [f for f in files if f.split('/')[2] in wanted]
        systems = sorted(wanted)

    out_root = resolve_out(args.out, 'mlaad_ptbr')
    print(f'  output: {out_root}')
    info(f'{len(systems)} TTS systems, {len(files)} files')

    per_system: Counter = Counter()
    written, skipped = 0, 0
    for path in sorted(files):
        system = path.split('/')[2]
        if path.endswith('.wav'):
            if args.max_per_system and per_system[system] >= args.max_per_system:
                continue
            per_system[system] += 1
        dest = out_root / path
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(REPO, path, repo_type='dataset', token=token)
        dest.write_bytes(Path(local).read_bytes())
        written += 1
        if (written + skipped) % 200 == 0:
            print(f'\r  {written} new / {skipped} existing, '
                  f'{len(per_system)} systems   ', end='', flush=True)
    print(flush=True)

    size = sum(f.stat().st_size for f in out_root.rglob('*') if f.is_file())
    info(f'{written} files written, {skipped} already present')
    for s in systems:
        info(f'  {s}: {per_system[s]} wavs')
    info(f'{human(size)} in {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
