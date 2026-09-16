"""
Download BRSpeech-DF bonafide samples from HuggingFace and organize by speaker.

Saves FLAC files to brspeech_df/{speaker_id}/{original_filename}
Skips speakers with fewer than --min_wavs files.

Usage:
    python download_brspeech_df.py
    python download_brspeech_df.py --out brspeech_df --min_wavs 5
"""
import argparse
import io
from pathlib import Path
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = 'AKCIT-Deepfake/BRSpeech-DF'


def shard_paths():
    files = list(list_repo_tree(REPO_ID, repo_type='dataset', path_in_repo='bonafide'))
    return sorted(f.path for f in files if f.path.endswith('.parquet'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='brspeech_df')
    ap.add_argument('--min_wavs', type=int, default=5)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)

    shards = shard_paths()
    print(f'{len(shards)} bonafide parquet shards to download', flush=True)

    # First pass: count per speaker across all shards to apply min_wavs filter
    # (each shard is typically one speaker so we can just process directly)

    total_saved = 0
    speaker_counts = defaultdict(int)

    for i, shard in enumerate(shards):
        print(f'[{i+1}/{len(shards)}] {shard} ...', flush=True)
        local = hf_hub_download(REPO_ID, shard, repo_type='dataset')
        df = pd.read_parquet(local)

        for _, row in df.iterrows():
            audio = row['audio']
            fname = Path(audio['path'])           # e.g. 10107_11436_000000-0001.flac
            spk   = fname.stem.split('_')[0]
            out_dir = out_root / spk
            out_dir.mkdir(exist_ok=True)
            dest = out_dir / fname.name
            if not dest.exists():
                dest.write_bytes(audio['bytes'])
            speaker_counts[spk] += 1

        print(f'  -> {len(df)} clips, speaker: {list(set(df["audio"].apply(lambda x: x["path"].split("_")[0])))[:3]}', flush=True)

    # Remove speakers below min_wavs
    removed = 0
    for spk, count in speaker_counts.items():
        if count < args.min_wavs:
            import shutil
            shutil.rmtree(out_root / spk, ignore_errors=True)
            removed += 1

    kept = len(speaker_counts) - removed
    total = sum(v for k, v in speaker_counts.items() if v >= args.min_wavs)
    print(f'\nDone. {kept} speakers, {total} clips -> {out_root}/')
    print(f'Removed {removed} speakers with < {args.min_wavs} clips.')


if __name__ == '__main__':
    main()
