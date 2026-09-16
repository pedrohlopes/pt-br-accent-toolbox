"""
Download Mozilla Common Voice (Portuguese) and organize it by speaker.

    common_voice/{client_id}/*.mp3

Mozilla moved the official distribution to the Mozilla Data Collective in October
2025, and the `mozilla-foundation/common_voice_*` HuggingFace repos no longer hold
audio. Two routes are supported:

  --source hf     (default) pull the per-split audio tars + transcript TSV from a
                  HuggingFace mirror (default: fsicoli/common_voice_17_0, which
                  keeps the original cv-corpus layout). Needs `huggingface_hub`.
  --source local  organize an already-downloaded cv-corpus tarball (cv_pt.tar.gz)
                  obtained from https://datacollective.mozillafoundation.org —
                  pass it with --archive.

Either way the speaker grouping comes from the TSV's `client_id`, and the
Brazilian variant filter (`variant == 'Portuguese (Brasil)'`) matches what the
accents project used: rows with a non-empty, non-Brazilian variant are dropped.

Usage:
    python tools/download_common_voice.py
    python tools/download_common_voice.py --split validated --min-clips 5
    python tools/download_common_voice.py --source local --archive /data/cv_pt.tar.gz
    python tools/download_common_voice.py --max-speakers 200 --max-clips 20
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import human, info, resolve_out, warn  # noqa: E402

DEFAULT_REPO = 'fsicoli/common_voice_17_0'
BR_VARIANT = 'Portuguese (Brasil)'


# ── transcripts ──────────────────────────────────────────────────────────────

def read_tsv(fp) -> list[dict]:
    csv.field_size_limit(10 ** 7)
    return list(csv.DictReader(io.TextIOWrapper(fp, encoding='utf-8'), delimiter='\t'))


def clip_index(rows, brazilian_only: bool) -> dict[str, str]:
    """clip filename -> client_id, after the variant filter."""
    index = {}
    for row in rows:
        variant = (row.get('variant') or '').strip()
        if brazilian_only and variant and variant != BR_VARIANT:
            continue
        index[row['path']] = row['client_id']
    return index


# ── sources ──────────────────────────────────────────────────────────────────

def hf_inputs(repo: str, split: str, out_root: Path):
    """(transcript rows, [audio tar paths]) from a HuggingFace mirror."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list(list_repo_files(repo, repo_type='dataset'))
    tsv = f'transcript/pt/{split}.tsv'
    if tsv not in files:
        raise SystemExit(f'{repo} has no {tsv} — available: '
                         f'{sorted(f for f in files if f.startswith("transcript/pt/"))}')
    tars = sorted(f for f in files if f.startswith(f'audio/pt/{split}/') and f.endswith('.tar'))
    if not tars:
        raise SystemExit(f'{repo} has no audio/pt/{split}/*.tar')

    print(f'  transcript: {tsv}', flush=True)
    tsv_path = hf_hub_download(repo, tsv, repo_type='dataset')
    with open(tsv_path, 'rb') as f:
        rows = read_tsv(f)

    print(f'  {len(tars)} audio tar(s) to fetch', flush=True)
    local = []
    for i, t in enumerate(tars, 1):
        print(f'  [{i}/{len(tars)}] {t}', flush=True)
        local.append(Path(hf_hub_download(repo, t, repo_type='dataset')))
    return rows, local


def local_inputs(archive: Path, split: str):
    """(transcript rows, [archive]) from a cv-corpus tarball on disk."""
    print(f'  reading {split}.tsv from {archive.name} ...', flush=True)
    with tarfile.open(archive, 'r:gz') as tf:
        member = next((m for m in tf.getmembers()
                       if m.name.endswith(f'pt/{split}.tsv')), None)
        if member is None:
            raise SystemExit(f'{archive} has no pt/{split}.tsv')
        rows = read_tsv(tf.extractfile(member))
    return rows, [archive]


# ── extraction ───────────────────────────────────────────────────────────────

def members(path: Path):
    """Iterate audio members of a .tar or .tar.gz, streaming."""
    mode = 'r|gz' if path.name.endswith(('.tar.gz', '.tgz')) else 'r|'
    with tarfile.open(path, mode=mode) as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(('.mp3', '.wav')):
                yield tf, m


def main():
    ap = argparse.ArgumentParser(description='Download Common Voice pt by speaker')
    ap.add_argument('--source', default='hf', choices=['hf', 'local'])
    ap.add_argument('--repo', default=DEFAULT_REPO,
                    help=f'HuggingFace mirror for --source hf (default: {DEFAULT_REPO})')
    ap.add_argument('--archive', default=None,
                    help='cv_pt.tar.gz for --source local')
    ap.add_argument('--split', default='validated',
                    help='validated | train | dev | test | other (default: validated)')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/common_voice)')
    ap.add_argument('--min-clips', type=int, default=1,
                    help='drop speakers with fewer than N clips')
    ap.add_argument('--max-clips', type=int, default=None,
                    help='keep at most N clips per speaker')
    ap.add_argument('--max-speakers', type=int, default=None)
    ap.add_argument('--all-variants', action='store_true',
                    help='keep European/African rows too (default: Brazilian only)')
    args = ap.parse_args()

    out_root = resolve_out(args.out, 'common_voice')
    print(f'=== Common Voice pt ({args.split}) ===')
    print(f'  output: {out_root}')

    if args.source == 'local':
        if not args.archive:
            raise SystemExit('--source local needs --archive /path/to/cv_pt.tar.gz')
        rows, archives = local_inputs(Path(args.archive), args.split)
    else:
        rows, archives = hf_inputs(args.repo, args.split, out_root)

    index = clip_index(rows, brazilian_only=not args.all_variants)
    per_speaker = Counter(index.values())
    keep_speakers = {s for s, n in per_speaker.items() if n >= args.min_clips}
    if args.max_speakers:
        keep_speakers = set(sorted(keep_speakers,
                                   key=lambda s: (-per_speaker[s], s))[:args.max_speakers])
    info(f'{len(index)} clips, {len(per_speaker)} speakers '
         f'-> {len(keep_speakers)} after filters')

    written, skipped = 0, 0
    kept = defaultdict(int)
    for archive in archives:
        for tf, m in members(archive):
            speaker = index.get(Path(m.name).name)
            if speaker is None or speaker not in keep_speakers:
                continue
            if args.max_clips and kept[speaker] >= args.max_clips:
                continue
            dest = out_root / speaker / Path(m.name).name
            if dest.exists() and dest.stat().st_size == m.size:
                skipped += 1
            else:
                src = tf.extractfile(m)
                if src is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read())
                written += 1
            kept[speaker] += 1
            if (written + skipped) % 2000 == 0:
                print(f'\r  {written} new / {skipped} existing, '
                      f'{len(kept)} speakers   ', end='', flush=True)
    print(flush=True)

    size = sum(f.stat().st_size for f in out_root.rglob('*') if f.is_file())
    info(f'{written} clips written, {skipped} already present')
    info(f'{len(kept)} speakers, {human(size)} in {out_root}')
    if not kept:
        warn('nothing extracted — check --split and the mirror layout')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
