"""
Download Certas Palavras from HuggingFace and organize it by speaker.

    https://huggingface.co/datasets/nilc-nlp/certas_palavras

Isolated-word reading, ~70 speakers, studio quality. Output:

    certas_palavras/
      metadata.csv          (filename, speaker_name, text)
      {speaker_name}/{split}_{i}.wav

Usage:
    python tools/download_certas_palavras.py
    python tools/download_certas_palavras.py --out /data/certas --splits train
    python tools/download_certas_palavras.py --max-per-speaker 40

Needs `datasets` and `soundfile`.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import human, info, resolve_out, warn  # noqa: E402

REPO = 'nilc-nlp/certas_palavras'
ALL_SPLITS = ('train', 'dev', 'test')


def safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)


def main():
    ap = argparse.ArgumentParser(description='Download Certas Palavras by speaker')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/certas_palavras)')
    ap.add_argument('--splits', default=','.join(ALL_SPLITS),
                    help='comma-separated splits to merge (default: all)')
    ap.add_argument('--max-per-speaker', type=int, default=None,
                    help='keep at most N clips per speaker')
    args = ap.parse_args()

    import soundfile as sf
    from datasets import Audio, load_dataset

    out_root = resolve_out(args.out, 'certas_palavras')
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    unknown = set(splits) - set(ALL_SPLITS)
    if unknown:
        warn(f'unknown splits: {sorted(unknown)} (known: {list(ALL_SPLITS)})')
        return 1

    print('=== Certas Palavras ===')
    print(f'  output: {out_root}')

    rows, per_speaker = [], Counter()
    for split in splits:
        print(f'  loading split: {split}', flush=True)
        ds = load_dataset(REPO, split=split)
        # Disable auto-decoding so we get raw bytes instead of requiring torchcodec
        ds = ds.cast_column('audio', Audio(decode=False))

        for i, sample in enumerate(ds):
            speaker = sample['speaker_name']
            if args.max_per_speaker and per_speaker[speaker] >= args.max_per_speaker:
                continue
            per_speaker[speaker] += 1

            speaker_dir = out_root / safe_name(speaker)
            speaker_dir.mkdir(parents=True, exist_ok=True)
            wav_path = speaker_dir / f'{split}_{i:06d}.wav'

            if not wav_path.exists():
                audio = sample['audio']   # {'bytes': ..., 'path': ...}
                raw = audio.get('bytes') or Path(audio['path']).read_bytes()
                array, sr = sf.read(io.BytesIO(raw))
                sf.write(str(wav_path), array, sr)

            rows.append({'filename': f'{speaker_dir.name}/{wav_path.name}',
                         'speaker_name': speaker, 'text': sample['text']})
            if (i + 1) % 500 == 0:
                print(f'\r  [{split}] {i + 1}/{len(ds)}   ', end='', flush=True)
        print(f'\r  [{split}] done ({len(ds)} samples)' + ' ' * 10, flush=True)

    csv_path = out_root / 'metadata.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'speaker_name', 'text'])
        writer.writeheader()
        writer.writerows(rows)

    size = sum(p.stat().st_size for p in out_root.rglob('*.wav'))
    info(f'metadata.csv ({len(rows)} samples) -> {csv_path}')
    info(f'{len(per_speaker)} speakers, {human(size)} in {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
