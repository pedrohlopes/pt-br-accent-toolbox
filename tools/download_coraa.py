"""
Download CORAA ASR v1.1 and split it into per-speaker folders.

CORAA (Candido Junior et al., LREC 2023) bundles several PT-BR sub-corpora in one
archive. Three of them carry usable speaker identity and are the ones this project
uses:

    coraa_nurc/{session_id}/*.wav    NURC-RE  — spontaneous interviews
    coraa_coral/{speaker_id}/*.wav   C-ORAL Brasil — Belo Horizonte / MG
    coraa_ted/{youtube_id}/*.wav     TEDx Talks

`alip` and `sp` are skipped: their filenames carry no speaker information.

Sources (see https://github.com/nilc-nlp/CORAA):
    train  — http://143.107.183.175:14888/static/coraa/train.zip   (59 GB; server refused connections
             in Sep 2026) — falls back to / can be forced to (--train-source hf) the 5-part RAR on HF
             gabrielrstan/CORAA-v1.1 train_dividido/train.part{1..5}.rar (53 GB; needs `unrar` or `7z`)
    dev    — HF gabrielrstan/CORAA-v1.1 dev.zip
    test   — HF gabrielrstan/CORAA-v1.1 test.zip
    metadata CSVs — same HF repo

The archives have no usable streaming index, so they are downloaded first
(resumable) and then extracted. Point --archive-dir at an existing download to
re-extract with different filters without re-fetching.

Usage:
    python tools/download_coraa.py --split dev
    python tools/download_coraa.py --split train --subsets nurc,coral
    python tools/download_coraa.py --split train --max-speakers 50 --max-utts 40
    python tools/download_coraa.py --split dev --archive-dir /data/downloads

Needs `huggingface_hub` for the dev/test splits and their metadata.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import http_download, human, info, resolve_out, warn  # noqa: E402

HF_REPO = 'gabrielrstan/CORAA-v1.1'
TRAIN_ZIP_URL = 'http://143.107.183.175:14888/static/coraa/train.zip'

SPLITS = {
    'train': {'zip': ('url', TRAIN_ZIP_URL), 'meta': 'metadata_train_final.csv'},
    'dev':   {'zip': ('hf', 'dev.zip'),      'meta': 'metadata_dev_final.csv'},
    'test':  {'zip': ('hf', 'test.zip'),     'meta': 'metadata_test_final.csv'},
}

TRAIN_RAR_PARTS = [f'train_dividido/train.part{i}.rar' for i in range(1, 6)]
SUBSETS = {'nurc': 'coraa_nurc', 'coral': 'coraa_coral', 'ted': 'coraa_ted'}

CORAL_RE = re.compile(r'_CO_(.+)\.wav$')
TED_RE = re.compile(r'^([A-Za-z0-9_\-]+)-\d+\.wav$')


def route(name: str):
    """Map an archive member to (subset_key, speaker, filename), or None."""
    parts = Path(name).parts
    if not name.endswith('.wav') or len(parts) < 3:
        return None
    sub = parts[1]

    if sub == 'NURC_RE':
        # {split}/NURC_RE/{type}/{session_id}/{file}.wav
        if len(parts) != 5:
            return None
        return 'nurc', parts[3], parts[4]

    if sub == 'CORAL':
        # {split}/CORAL/{id}_CO_{speaker_id}.wav
        m = CORAL_RE.search(parts[2])
        return ('coral', m.group(1), parts[2]) if m else None

    if sub.startswith('Ted_part'):
        # {split}/Ted_partX/{yt_id}-{seg}.wav
        m = TED_RE.match(parts[2])
        return ('ted', m.group(1), parts[2]) if m else None

    return None  # alip, sp — no speaker info


def fetch_zip(split: str, archive_dir: Path) -> Path:
    kind, ref = SPLITS[split]['zip']
    if kind == 'url':
        return http_download(ref, archive_dir / f'coraa_{split}.zip')
    from huggingface_hub import hf_hub_download
    print(f'  hf_hub_download({HF_REPO}, {ref})', flush=True)
    return Path(hf_hub_download(HF_REPO, ref, repo_type='dataset',
                                local_dir=str(archive_dir)))


def fetch_train_rar(archive_dir: Path) -> Path:
    """Train split from the HF multipart RAR: download the 5 parts (resumable), extract the whole split into
    archive_dir/coraa_train_extracted/ (route() then keeps only the wanted sub-corpora) and return that dir."""
    out = archive_dir / 'coraa_train_extracted'
    marker = out / '.extracted'
    if marker.exists():                         # a previous run finished extracting: RARs not needed
        info(f'already extracted -> {out}')
        return out
    from huggingface_hub import hf_hub_download
    parts = []
    for p in TRAIN_RAR_PARTS:
        print(f'  hf_hub_download({HF_REPO}, {p})', flush=True)
        parts.append(Path(hf_hub_download(HF_REPO, p, repo_type='dataset', local_dir=str(archive_dir))))
    out.mkdir(parents=True, exist_ok=True)
    if shutil.which('unrar'):
        cmd = ['unrar', 'x', '-o+', '-idq', str(parts[0]), str(out) + '/']
    elif shutil.which('7z'):
        cmd = ['7z', 'x', '-y', f'-o{out}', str(parts[0])]
    else:
        raise RuntimeError('extracting the CORAA train RAR needs `unrar` or `7z` on PATH')
    print(f'  extracting {parts[0].name} (+{len(parts) - 1} parts) -> {out}', flush=True)
    subprocess.run(cmd, check=True)
    marker.touch()
    return out


def _split_root(src: Path, max_depth: int = 3) -> Path:
    """Directory (<= max_depth below src) whose children are the sub-corpus folders (NURC_RE, CORAL,
    Ted_part*), so extracted layouts with or without a top-level {split}/ folder both route correctly."""
    level = [src]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            kids = [c for c in d.iterdir() if c.is_dir()]
            names = {c.name for c in kids}
            if names & {'NURC_RE', 'CORAL'} or any(n.startswith('Ted_part') for n in names):
                return d
            nxt += kids
        level = nxt
    raise RuntimeError(f'no NURC_RE / CORAL / Ted_part* folder found under {src}')


def members(src: Path, split: str = 'train'):
    """(name, size, read, path) for every file of a zip archive or an extracted directory; `name` is the
    archive-relative posix path that route() expects ({split}/{subcorpus}/...); `path` is the extracted file
    (None for zip members) so it can be hard-linked instead of copied."""
    if src.is_dir():
        root = _split_root(src)
        for f in sorted(root.rglob('*.wav')):
            yield f'{split}/{f.relative_to(root).as_posix()}', f.stat().st_size, f.read_bytes, f
        return
    with zipfile.ZipFile(src) as zf:
        for m in zf.infolist():
            yield m.filename, m.file_size, (lambda m=m: zf.read(m)), None


def fetch_metadata(split: str, out_root: Path):
    name = SPLITS[split]['meta']
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, name, repo_type='dataset',
                               local_dir=str(out_root))
        info(f'metadata -> {path}')
    except Exception as e:                       # non-fatal: audio still usable
        warn(f'could not fetch {name}: {e}')


def main():
    ap = argparse.ArgumentParser(description='Download + split CORAA ASR v1.1')
    ap.add_argument('--split', default='train', choices=sorted(SPLITS))
    ap.add_argument('--subsets', default='nurc,coral,ted',
                    help='comma-separated: nurc, coral, ted (default: all three)')
    ap.add_argument('--out', default=None,
                    help='parent directory for coraa_* folders (default: ACCENTS_BASE)')
    ap.add_argument('--archive-dir', default=None,
                    help='where the zip lives / is downloaded (default: --out)')
    ap.add_argument('--max-speakers', type=int, default=None,
                    help='keep at most N speakers per subset')
    ap.add_argument('--max-utts', type=int, default=None,
                    help='keep at most N utterances per speaker')
    ap.add_argument('--delete-archive', action='store_true',
                    help='remove the zip once extraction finishes')
    ap.add_argument('--no-metadata', action='store_true')
    ap.add_argument('--train-source', default='auto', choices=['auto', 'url', 'hf'],
                    help='train split: original zip URL, the HF multipart RAR, or auto (URL, then HF on failure)')
    args = ap.parse_args()

    wanted = {s.strip() for s in args.subsets.split(',') if s.strip()}
    unknown = wanted - set(SUBSETS)
    if unknown:
        warn(f'unknown subsets: {sorted(unknown)} (known: {sorted(SUBSETS)})')
        return 1

    out_root = resolve_out(args.out, '')
    archive_dir = Path(args.archive_dir) if args.archive_dir else out_root
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== CORAA {args.split} ({", ".join(sorted(wanted))}) ===')
    print(f'  output: {out_root}')

    if not args.no_metadata:
        fetch_metadata(args.split, out_root)

    if args.split == 'train' and args.train_source == 'hf':
        zip_path = fetch_train_rar(archive_dir)
    elif args.split == 'train' and args.train_source == 'auto':
        try:
            zip_path = fetch_zip(args.split, archive_dir)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            warn(f'train zip URL failed ({e}); using the HF multipart RAR instead')
            zip_path = fetch_train_rar(archive_dir)
    else:
        zip_path = fetch_zip(args.split, archive_dir)

    dests = {k: out_root / SUBSETS[k] for k in wanted}
    for d in dests.values():
        d.mkdir(parents=True, exist_ok=True)

    speakers: dict[str, dict[str, int]] = {k: {} for k in wanted}
    counts, skipped = Counter(), 0

    print(f'  extracting {zip_path.name} ...', flush=True)
    for name, size, read, path in members(zip_path, args.split):
        hit = route(name)
        if not hit:
            continue
        key, speaker, filename = hit
        if key not in wanted:
            continue

        seen = speakers[key]
        if speaker not in seen:
            if args.max_speakers and len(seen) >= args.max_speakers:
                continue
            seen[speaker] = 0
        if args.max_utts and seen[speaker] >= args.max_utts:
            continue

        dest = dests[key] / speaker / filename
        if dest.exists() and dest.stat().st_size == size:
            skipped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:                                    # extracted tree: hard link (no extra I/O or space)
                if path is None:
                    raise OSError
                dest.unlink(missing_ok=True)
                os.link(path, dest)
            except OSError:
                dest.write_bytes(read())
            counts[key] += 1
        seen[speaker] += 1

        done = sum(counts.values()) + skipped
        if done % 5000 == 0:
            print(f'\r  {done} files  ' +
                  '  '.join(f'{k}={counts[k]}' for k in sorted(wanted)) + '   ',
                  end='', flush=True)
    print(flush=True)

    if args.delete_archive:
        if zip_path.is_dir():
            shutil.rmtree(zip_path, ignore_errors=True)
        else:
            zip_path.unlink(missing_ok=True)
        info(f'removed {zip_path.name}')
    else:
        info(f'archive kept at {zip_path} — reuse it with --archive-dir')

    for key in sorted(wanted):
        d = dests[key]
        size = sum(f.stat().st_size for f in d.rglob('*.wav'))
        info(f'{SUBSETS[key]}: {counts[key]} new files, '
             f'{len(speakers[key])} speakers, {human(size)} -> {d}')
    if skipped:
        info(f'{skipped} files already present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
