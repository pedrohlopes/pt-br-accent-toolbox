"""
Download the Sotaque Brasileiro accent corpus.

    https://github.com/sotaque-brasileiro/sotaque-brasileiro  (Milan, 2021)
    DOI 10.5281/zenodo.5466945

Crowdsourced PT-BR recordings with rich speaker metadata — birth city/state,
current city/state, age, gender — which is what makes it usable as a regional
accent corpus. The project publishes daily snapshots as GitHub release assets;
this script grabs the most recent one (~1.6 GB) and unpacks it to:

    sotaque_brasileiro/sotaque-brasileiro-data/
        sotaque-brasileiro.csv      one row per recording
        accent/{uuid}.wav

With --by-state it additionally builds a per-class tree of hardlinks (falling back
to copies), which is the layout the accent classifiers expect:

    sotaque_brasileiro/by_state/{UF}/{uuid}.wav

Usage:
    python tools/download_sotaque_brasileiro.py
    python tools/download_sotaque_brasileiro.py --by-state --min-per-state 20
    python tools/download_sotaque_brasileiro.py --group-by birth_city
    python tools/download_sotaque_brasileiro.py --release 2023-05-02

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import UA, http_download, human, info, resolve_out, warn  # noqa: E402

API = 'https://api.github.com/repos/sotaque-brasileiro/sotaque-brasileiro/releases'


def latest_asset(tag: str | None) -> tuple[str, str, int]:
    """(tag, download_url, size) of the newest (or named) release's zip asset."""
    url = f'{API}/tags/{tag}' if tag else f'{API}?per_page=10'
    headers = dict(UA, Accept='application/vnd.github+json')
    gh_token = os.environ.get('GITHUB_TOKEN')
    if gh_token:
        headers['Authorization'] = f'Bearer {gh_token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    releases = [payload] if tag else payload
    for rel in releases:
        for asset in rel.get('assets', []):
            if asset['name'].endswith('.zip'):
                return rel['tag_name'], asset['browser_download_url'], asset['size']
    raise SystemExit('no zip asset found in the release listing')


def group_key(row: dict, field: str) -> str | None:
    value = (row.get(field) or '').strip()
    return value or None


def main():
    ap = argparse.ArgumentParser(description='Download Sotaque Brasileiro')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/sotaque_brasileiro)')
    ap.add_argument('--release', default=None,
                    help='release tag, e.g. 2023-05-02 (default: newest)')
    ap.add_argument('--by-state', action='store_true',
                    help='also build by_state/{UF}/*.wav from the metadata CSV')
    ap.add_argument('--group-by', default='birth_state',
                    help='metadata column used by --by-state (default: birth_state)')
    ap.add_argument('--min-per-state', type=int, default=0,
                    help='with --by-state, drop classes with fewer than N recordings')
    ap.add_argument('--delete-archive', action='store_true')
    args = ap.parse_args()

    out_root = resolve_out(args.out, 'sotaque_brasileiro')
    print('=== Sotaque Brasileiro ===')
    print(f'  output: {out_root}')

    tag, url, size = latest_asset(args.release)
    info(f'release {tag} ({human(size)})')
    archive = http_download(url, out_root / Path(url).name)

    print(f'  extracting {archive.name} ...', flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_root)
    if args.delete_archive:
        archive.unlink(missing_ok=True)
        info(f'removed {archive.name}')

    csv_paths = sorted(out_root.rglob('sotaque-brasileiro.csv'))
    if not csv_paths:
        warn('no sotaque-brasileiro.csv found in the archive')
        return 1
    meta_path = csv_paths[0]
    data_root = meta_path.parent
    n_wavs = sum(1 for _ in data_root.rglob('*.wav'))
    info(f'{n_wavs} wavs, metadata at {meta_path}')

    if not args.by_state:
        return 0

    with open(meta_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if args.group_by not in (rows[0] if rows else {}):
        warn(f'column "{args.group_by}" not in the CSV; available: '
             f'{sorted(rows[0]) if rows else []}')
        return 1

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = group_key(row, args.group_by)
        if key:
            by_class[key].append(row)

    dest_root = out_root / f'by_{args.group_by}'
    dest_root.mkdir(parents=True, exist_ok=True)
    linked, missing, dropped = 0, 0, Counter()
    for key, items in sorted(by_class.items()):
        if len(items) < args.min_per_state:
            dropped[key] = len(items)
            continue
        safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in key)
        class_dir = dest_root / safe
        class_dir.mkdir(exist_ok=True)
        for row in items:
            src = data_root / row['audio_file_path']
            if not src.exists():
                missing += 1
                continue
            dest = class_dir / src.name
            if dest.exists():
                continue
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
            linked += 1

    info(f'{len(by_class) - len(dropped)} classes, {linked} recordings -> {dest_root}')
    if missing:
        warn(f'{missing} rows pointed at wavs that are not in the archive')
    if dropped:
        info(f'dropped {len(dropped)} classes below --min-per-state '
             f'({", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
