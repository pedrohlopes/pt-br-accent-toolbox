"""
Download the NURC-SP Audio Corpus (CORAA) from HuggingFace.

    https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus

Repo layout:
    train.tar.gz / dev.tar.gz / test.tar.gz
    original/audios_{split}_metadata.csv   full-length session audio
    filtered/audios_{split}_metadata.csv   cleaned/segmented version

Output:
    nurcsp/{split}/...                     as laid out inside the tarball
    nurcsp/{original,filtered}/*.csv       metadata

NURC-SP is spontaneous São Paulo speech (interviews and dialogues); the project
uses it for accent-switching and prosody experiments, where the `filtered`
metadata is the more useful of the two.

Usage:
    python tools/download_nurcsp.py --splits dev,test
    python tools/download_nurcsp.py --splits train
    python tools/download_nurcsp.py --metadata-only

Needs `huggingface_hub`.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import human, info, resolve_out, warn  # noqa: E402

REPO = 'nilc-nlp/CORAA-NURC-SP-Audio-Corpus'
SPLITS = ('train', 'dev', 'test')
META_KINDS = ('original', 'filtered')


def main():
    ap = argparse.ArgumentParser(description='Download the NURC-SP audio corpus')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/nurcsp)')
    ap.add_argument('--splits', default='dev,test',
                    help='comma-separated: train, dev, test (train is ~40 GB)')
    ap.add_argument('--metadata-only', action='store_true')
    ap.add_argument('--keep-archive', action='store_true',
                    help='keep the downloaded tar.gz files')
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    out_root = resolve_out(args.out, 'nurcsp')
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    unknown = set(splits) - set(SPLITS)
    if unknown:
        warn(f'unknown splits: {sorted(unknown)} (known: {list(SPLITS)})')
        return 1

    print('=== NURC-SP (CORAA) ===')
    print(f'  output: {out_root}')

    for kind in META_KINDS:
        for split in SPLITS:
            name = f'{kind}/audios_{split}_metadata.csv'
            try:
                path = hf_hub_download(REPO, name, repo_type='dataset',
                                       local_dir=str(out_root))
                info(f'{name} -> {path}')
            except Exception as e:
                warn(f'{name}: {e}')

    if args.metadata_only:
        return 0

    for split in splits:
        print(f'  fetching {split}.tar.gz ...', flush=True)
        archive = Path(hf_hub_download(REPO, f'{split}.tar.gz', repo_type='dataset',
                                       local_dir=str(out_root)))
        dest = out_root / split
        dest.mkdir(parents=True, exist_ok=True)
        print(f'  extracting -> {dest}', flush=True)
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(dest, filter='data')
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            info(f'removed {archive.name}')
        n = sum(1 for _ in dest.rglob('*.wav'))
        info(f'{split}: {n} wavs in {dest}')

    size = sum(f.stat().st_size for f in out_root.rglob('*') if f.is_file())
    info(f'{human(size)} in {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
