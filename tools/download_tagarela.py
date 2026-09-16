"""
Download TAGARELA podcast audio, optionally restricted to a list of episodes.

    https://huggingface.co/datasets/freds0/TAGARELA

TAGARELA is ~1764 parquet shards of segmented Brazilian podcast speech. Paths look
like:

    podcasts_segmented_enhanced_vocos_16khz_flac/{n}/{L}/show_{show}/{ep}_{hash}_{t0}_{t1}-{seg}.flac

where {ep} is the 22-character Spotify episode ID, which is what this project uses
as the speaker/session key:

    tagarela/{episode_id}/*.wav

Two modes:

  * Episode-filtered (--episodes): a two-phase run — scan every shard reading only
    the `path` column (cheap HTTP range reads) to find which shards hold the target
    episodes, then re-open only those and write their audio. This is how the
    project's Spotify subset was built; supply your own ID list.
  * Unfiltered (default): walk shards in order and take everything until
    --max-episodes / --max-shards is hit. Good for a quick sample of the corpus.

Usage:
    python tools/download_tagarela.py --max-shards 2 --max-episodes 10
    python tools/download_tagarela.py --episodes @spotify_ids.txt
    python tools/download_tagarela.py --episodes 4tXNAuq0AlZKqVuOmSNsoP --jobs 4

--episodes takes a comma-separated list or @file (one ID per line; a CSV column of
IDs also works — anything that looks like a 22-char base62 token is picked up).

Needs `pandas`, `pyarrow`, `fsspec`, `soundfile`.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import threading
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import human, info, resolve_out, warn  # noqa: E402

HF_BASE = 'https://huggingface.co/datasets/freds0/TAGARELA/resolve/main'
NUM_SHARDS = 1764
EPISODE_RE = re.compile(r'/([A-Za-z0-9]{22})_[0-9a-f]{40}_')
ID_RE = re.compile(r'\b([A-Za-z0-9]{22})\b')

lock = threading.Lock()


def shard_url(i: int) -> str:
    return f'{HF_BASE}/data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet'


def open_parquet(url: str, retries: int = 5):
    """Open a remote parquet file, backing off on HF rate limits."""
    import fsspec
    import pyarrow.parquet as pq
    delay = 10
    for attempt in range(retries):
        try:
            fs, path = fsspec.url_to_fs(url)
            return pq.ParquetFile(fs.open(path))
        except Exception as e:
            if attempt == retries - 1:
                raise
            if '429' in str(e) or 'Too Many' in str(e):
                with lock:
                    print(f'  rate-limited, waiting {delay}s...', flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise


def parse_episodes(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    if spec.startswith('@'):
        spec = Path(spec[1:]).read_text(encoding='utf-8')
    ids = set(ID_RE.findall(spec))
    if not ids:
        raise SystemExit('no 22-character episode IDs found in --episodes')
    return ids


def write_wav(dest: Path, raw: bytes):
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(raw), dtype='int16')
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes() if audio.ndim == 1 else audio.flatten().tobytes())
    dest.write_bytes(buf.getvalue())


# ── phase 1: which shards hold the target episodes ───────────────────────────

def scan_shard(i: int, targets: set[str]):
    hits = []
    try:
        pf = open_parquet(shard_url(i))
        offset = 0
        for rg in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(rg, columns=['path'])
            for j, p in enumerate(batch['path'].to_pylist()):
                m = EPISODE_RE.search(p)
                if m and m.group(1) in targets:
                    hits.append(offset + j)
            offset += batch.num_rows
    except Exception as e:
        with lock:
            print(f'  [scan] shard {i}: {e}', flush=True)
    return i, hits


# ── phase 2: pull the audio ──────────────────────────────────────────────────

def extract_shard(i: int, out_root: Path, rows: set[int] | None,
                  targets: set[str] | None, counters: Counter,
                  max_episodes: int | None, seen: set[str],
                  max_per_episode: int | None, per_episode: Counter):
    saved = 0
    try:
        pf = open_parquet(shard_url(i))
        offset = 0
        for rg in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(rg, columns=['audio', 'path'])
            paths = batch['path'].to_pylist()
            audios = batch['audio'].to_pylist()
            for j, (p, blob) in enumerate(zip(paths, audios)):
                if rows is not None and (offset + j) not in rows:
                    continue
                m = EPISODE_RE.search(p)
                if not m:
                    continue
                ep = m.group(1)
                if targets is not None and ep not in targets:
                    continue
                with lock:
                    if ep not in seen:
                        if max_episodes and len(seen) >= max_episodes:
                            continue
                        seen.add(ep)
                    if max_per_episode and per_episode[ep] >= max_per_episode:
                        continue
                    per_episode[ep] += 1

                dest = out_root / ep / (Path(p).stem + '.wav')
                if dest.exists():
                    with lock:
                        counters['skipped'] += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_wav(dest, blob['bytes'])
                saved += 1
            offset += len(paths)
    except Exception as e:
        with lock:
            print(f'  [audio] shard {i}: {e}', flush=True)
    with lock:
        counters['saved'] += saved
        print(f'  shard {i}: +{saved} (total {counters["saved"]})', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Download TAGARELA podcast audio')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/tagarela)')
    ap.add_argument('--episodes', default=None,
                    help='comma-separated 22-char Spotify episode IDs, or @file')
    ap.add_argument('--max-episodes', type=int, default=None,
                    help='stop after N distinct episodes (unfiltered mode)')
    ap.add_argument('--max-per-episode', type=int, default=None,
                    help='keep at most N segments per episode')
    ap.add_argument('--max-shards', type=int, default=None,
                    help='only look at the first N shards (unfiltered mode)')
    ap.add_argument('--jobs', type=int, default=2,
                    help='parallel shard workers — stay low, HF rate-limits (default: 2)')
    ap.add_argument('--scan-jobs', type=int, default=4,
                    help='parallel workers for the episode scan phase (default: 4)')
    args = ap.parse_args()

    out_root = resolve_out(args.out, 'tagarela')
    targets = parse_episodes(args.episodes)
    print('=== TAGARELA ===')
    print(f'  output: {out_root}')

    counters: Counter = Counter()
    seen: set[str] = set()
    per_episode: Counter = Counter()

    if targets:
        info(f'{len(targets)} target episodes')
        print(f'  phase 1: scanning {NUM_SHARDS} shards '
              f'({args.scan_jobs} workers, path column only)...', flush=True)
        matching: dict[int, set[int]] = {}
        scanned = 0
        with ThreadPoolExecutor(max_workers=args.scan_jobs) as ex:
            futs = [ex.submit(scan_shard, i, targets) for i in range(NUM_SHARDS)]
            for fut in as_completed(futs):
                i, hits = fut.result()
                scanned += 1
                if hits:
                    matching[i] = set(hits)
                    print(f'  hit shard {i} ({len(hits)} rows)', flush=True)
                elif scanned % 100 == 0:
                    print(f'  scanned {scanned}/{NUM_SHARDS}, '
                          f'{len(matching)} matching shards', flush=True)
        info(f'{len(matching)} shards hold {sum(len(v) for v in matching.values())} rows')
        shards = sorted(matching)
        rows_for = matching
    else:
        shards = list(range(args.max_shards or NUM_SHARDS))
        rows_for = {}
        info(f'unfiltered mode: walking {len(shards)} shard(s)')

    print(f'  phase 2: extracting audio ({args.jobs} workers)...', flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(extract_shard, i, out_root, rows_for.get(i), targets,
                          counters, args.max_episodes, seen,
                          args.max_per_episode, per_episode)
                for i in shards]
        for fut in as_completed(futs):
            fut.result()
            if args.max_episodes and len(seen) >= args.max_episodes and not targets:
                break

    n_ep = sum(1 for p in out_root.iterdir() if p.is_dir())
    size = sum(f.stat().st_size for f in out_root.rglob('*.wav'))
    info(f'{counters["saved"]} saved, {counters["skipped"]} already present')
    info(f'{n_ep} episodes, {human(size)} in {out_root}')
    if not counters['saved'] and not counters['skipped']:
        warn('nothing downloaded — check --episodes')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
