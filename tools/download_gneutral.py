"""
Download GneutralSpeech datasets (male + female) from Kaggle and organise them.

Datasets (Globo / MediaTechLab):
  Male:   mediatechlab/gneutralspeech          (~20 h, single male speaker)
  Female: mediatechlab/g-neutral-speech-female  (~20 h, single female speaker)

Output:
  gneutral_wavs/gneutral_male/   *.wav  (16 kHz mono)
  gneutral_wavs/gneutral_female/ *.wav  (16 kHz mono)

Requirements:
  pip install kaggle soundfile librosa
  kaggle.json credentials in ~/.kaggle/kaggle.json (chmod 600)

Usage:
  python download_gneutral.py
  python download_gneutral.py --max-wavs 500  # keep only N wavs per speaker
"""
import argparse
import io
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

BASE    = Path(__file__).parent
OUT_DIR = BASE / 'gneutral_wavs'
SR_OUT  = 16_000

DATASETS = [
    ('gneutral_male',   'mediatechlab/gneutralspeech'),
    ('gneutral_female', 'mediatechlab/g-neutral-speech-female'),
]


def download_zip(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f'  kaggle datasets download -d {slug} -p {dest}')
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', slug, '-p', str(dest)],
        check=True,
    )
    zips = sorted(dest.glob('*.zip'))
    if not zips:
        raise FileNotFoundError(f'No zip found in {dest} after download')
    return zips[0]


def resample_to_16k(array: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR_OUT:
        return array.astype(np.float32)
    import librosa
    return librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=SR_OUT)


def extract_wavs(zip_path: Path, out_dir: Path, max_wavs: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob('*.wav'))
    if existing:
        print(f'  {out_dir.name}: {len(existing)} wavs already present — skipping extraction')
        return

    print(f'  Scanning {zip_path.name}...')
    with zipfile.ZipFile(zip_path) as zf:
        wav_names = [n for n in zf.namelist() if n.lower().endswith('.wav')]

    print(f'  Found {len(wav_names)} wav files inside zip')
    if max_wavs and len(wav_names) > max_wavs:
        rng = random.Random(42)
        wav_names = sorted(rng.sample(wav_names, max_wavs))
        print(f'  Sampling {max_wavs} wavs (seed=42)')

    print(f'  Extracting & resampling → {out_dir}')
    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(wav_names):
            raw = zf.read(name)
            try:
                audio, sr = sf.read(io.BytesIO(raw))
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = resample_to_16k(audio, sr)
            stem = Path(name).stem
            sf.write(str(out_dir / f'{stem}.wav'), audio, SR_OUT, subtype='PCM_16')
            if (i + 1) % 500 == 0:
                print(f'    {i+1}/{len(wav_names)}')

    print(f'  Done — {len(list(out_dir.glob("*.wav")))} wavs in {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-wavs', type=int, default=50,
                    help='Keep at most this many wavs per speaker (random subset)')
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    for name, slug in DATASETS:
        print(f'\n=== {name} ({slug}) ===')
        tmp = OUT_DIR / f'_download_{name}'
        out = OUT_DIR / name

        if list(out.glob('*.wav')):
            print(f'  Already have wavs in {out} — skipping download')
            continue

        zip_path = download_zip(slug, tmp)
        extract_wavs(zip_path, out, args.max_wavs)

        # Clean up zip download temp dir
        shutil.rmtree(tmp, ignore_errors=True)

    print('\nAll done.')
    for name, _ in DATASETS:
        d = OUT_DIR / name
        n = len(list(d.glob('*.wav'))) if d.exists() else 0
        print(f'  {name}: {n} wavs')


if __name__ == '__main__':
    main()
