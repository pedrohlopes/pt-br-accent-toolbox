"""
Download Portuguese YODAS samples from HuggingFace.

The dataset (AdoCleanCode/portuguese_yodas_mfa_aligned) has 963 samples
from Brazilian Portuguese YouTube videos. Speaker IDs are all 'unknown_speaker',
so samples are organized by their original sample_id.

Saves WAV files to accents/yodas_samples/{sample_id}.wav

Usage:
    python download_yodas.py
    python download_yodas.py --out yodas_samples --max_samples 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import resolve_out  # noqa: E402

import pandas as pd
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = 'AdoCleanCode/portuguese_yodas_mfa_aligned'
PARQUET_PATH = 'data/train-00000-of-00001.parquet'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/yodas_samples)')
    ap.add_argument('--max_samples', type=int, default=None,
                    help='Max samples to download (None = all)')
    args = ap.parse_args()

    out_root = resolve_out(args.out, 'yodas_samples')

    # Download parquet
    print(f'Downloading parquet from {REPO_ID} ...', flush=True)
    local = hf_hub_download(REPO_ID, PARQUET_PATH, repo_type='dataset')
    df = pd.read_parquet(local)
    print(f'Loaded {len(df)} samples', flush=True)

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f'Limiting to {args.max_samples} samples', flush=True)

    saved = 0
    skipped = 0
    for i, row in df.iterrows():
        sid = row['id']
        dest = out_root / f'{sid}.wav'
        if dest.exists():
            skipped += 1
            continue

        audio = row['audio']
        samples = np.frombuffer(audio['bytes'], dtype=np.int16)
        sf.write(dest, samples, samplerate=16000)
        saved += 1

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(df)}] saved={saved} skipped={skipped}', flush=True)

    print(f'\nDone. saved={saved} skipped={skipped} total={saved+skipped} -> {out_root}/')


if __name__ == '__main__':
    main()
