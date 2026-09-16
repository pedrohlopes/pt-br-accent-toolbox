"""
Download CML-TTS Portuguese and flatten it to per-speaker folders.

CML-TTS (Oliveira et al., TSD 2023) is a multilingual read-speech corpus derived
from LibriVox. The Portuguese release is a single 9.7 GB tar.bz on OpenSLR:

    https://www.openslr.org/146/  ->  cml_tts_dataset_portuguese_v0.1.tar.bz

Archive layout:
    cml_tts_dataset_portuguese_v0.1/{split}/audio/{speaker}/{chapter}/{file}.wav

Output (splits merged, chapter level dropped):
    cml_tts/{speaker_id}/*.wav

The tar.bz is streamed by default, so only the wavs that survive the filters hit
the disk. --keep-archive downloads it first instead (resumable), which is what you
want if you plan to re-filter later.

Usage:
    python tools/download_cml_tts.py
    python tools/download_cml_tts.py --max-speakers 50 --max-utts 40
    python tools/download_cml_tts.py --splits train --keep-archive
    python tools/download_cml_tts.py --archive /data/cml_tts_pt.tar.bz   # local file

Stdlib only.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import UA, http_download, human, info, resolve_out, warn  # noqa: E402

import urllib.request  # noqa: E402

URL = 'https://www.openslr.org/resources/146/cml_tts_dataset_portuguese_v0.1.tar.bz'
MIRROR = 'https://openslr.magicdatatech.com/resources/146/cml_tts_dataset_portuguese_v0.1.tar.bz'


def route(name: str, splits: set[str] | None):
    """{root}/{split}/audio/{speaker}/{chapter}/{file}.wav -> (speaker, filename)."""
    parts = Path(name).parts
    if len(parts) < 6 or not name.endswith('.wav') or parts[2] != 'audio':
        return None
    split, speaker, filename = parts[1], parts[3], parts[5]
    if splits and split not in splits:
        return None
    return speaker, filename


def iter_members(path_or_url: str, local: Path | None):
    if local is not None:
        with tarfile.open(local, 'r:bz2') as tf:
            for m in tf:
                if m.isfile():
                    yield tf, m
        return
    req = urllib.request.Request(path_or_url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        with tarfile.open(fileobj=r, mode='r|bz2') as tf:
            for m in tf:
                if m.isfile():
                    yield tf, m


def main():
    ap = argparse.ArgumentParser(description='Download CML-TTS Portuguese by speaker')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/cml_tts)')
    ap.add_argument('--splits', default='train,dev,test',
                    help='comma-separated splits to merge (default: all)')
    ap.add_argument('--max-speakers', type=int, default=None)
    ap.add_argument('--max-utts', type=int, default=None,
                    help='keep at most N wavs per speaker')
    ap.add_argument('--archive', default=None,
                    help='use this local tar.bz instead of downloading')
    ap.add_argument('--keep-archive', action='store_true',
                    help='download the tarball to disk (resumable) instead of streaming')
    ap.add_argument('--archive-dir', default=None)
    ap.add_argument('--mirror', action='store_true',
                    help='use the OpenSLR CN mirror')
    args = ap.parse_args()

    out_root = resolve_out(args.out, 'cml_tts')
    splits = {s.strip() for s in args.splits.split(',') if s.strip()}
    url = MIRROR if args.mirror else URL

    print('=== CML-TTS Portuguese ===')
    print(f'  output: {out_root}')
    print(f'  splits: {", ".join(sorted(splits))}')

    local = Path(args.archive) if args.archive else None
    if local is None and args.keep_archive:
        archive_dir = Path(args.archive_dir) if args.archive_dir else out_root
        local = http_download(url, archive_dir / Path(url).name)
    if local is None:
        print(f'  streaming {url}', flush=True)

    def quotas_full() -> bool:
        """Nothing more can match: speaker slots full and every one of them capped."""
        return bool(args.max_speakers and args.max_utts
                    and len(per_speaker) >= args.max_speakers
                    and all(n >= args.max_utts for n in per_speaker.values()))

    per_speaker: Counter = Counter()
    written, skipped = 0, 0
    for tf, m in iter_members(url, local):
        hit = route(m.name, splits)
        if not hit:
            continue
        speaker, filename = hit
        if speaker not in per_speaker:
            if args.max_speakers and len(per_speaker) >= args.max_speakers:
                if quotas_full():
                    print('\r  quotas reached — stopping the stream early' + ' ' * 20,
                          flush=True)
                    break
                continue
            per_speaker[speaker] = 0
        if args.max_utts and per_speaker[speaker] >= args.max_utts:
            if quotas_full():
                print('\r  quotas reached — stopping the stream early' + ' ' * 20,
                      flush=True)
                break
            continue

        dest = out_root / speaker / filename
        if dest.exists() and dest.stat().st_size == m.size:
            skipped += 1
        else:
            src = tf.extractfile(m)
            if src is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read())
            written += 1
        per_speaker[speaker] += 1
        if (written + skipped) % 1000 == 0:
            print(f'\r  {written} new / {skipped} existing, '
                  f'{len(per_speaker)} speakers   ', end='', flush=True)
    print(flush=True)

    size = sum(f.stat().st_size for f in out_root.rglob('*.wav'))
    info(f'{written} wavs written, {skipped} already present')
    info(f'{len(per_speaker)} speakers, {human(size)} in {out_root}')
    if not per_speaker:
        warn('nothing extracted — check --splits')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
