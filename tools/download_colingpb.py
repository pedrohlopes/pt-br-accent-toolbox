"""
Download CoLingPB interview WAVs and diarize them into 2-speaker segments.

CoLingPB (Corpus Linguístico da Paraíba) is published as a DSpace community in
the UFPB institutional repository, one item per informant:

    https://repositorio.ufpb.br/jspui/handle/123456789/23184
    CoLingPB/{region}/{city}/{informant}   e.g. CoLingPB/Mata Rural/João Pessoa/brPB01_g1bF01

This script discovers those items itself — browsing the community and reading each
item's WAV bitstream — so nothing has to be prepared by hand. The manifest is
cached as `colingpb_urls.json` next to the audio; later runs reuse it unless you
pass --refresh.

Pipeline per recording:
  1. Download the WAV from repositorio.ufpb.br
  2. Resample to 16 kHz mono
  3. pyannote/speaker-diarization-3.1 -> speaker timeline
  4. Merge adjacent segments, discard anything under MIN_SEG_S
  5. Save as colingpb/{informant}_spkA/{informant}_{seg:04d}.wav
                  colingpb/{informant}_spkB/{informant}_{seg:04d}.wav

Each recording is an interview, so speaker A/B are the informant and the
interviewer — which of the two is which is not recorded upstream and this script
does not guess.

Usage:
    huggingface-cli login           # or export HF_TOKEN; needs access to pyannote/speaker-diarization-3.1
    python tools/download_colingpb.py
    python tools/download_colingpb.py --discover-only        # just write the manifest
    python tools/download_colingpb.py --regions "Mata Urbana,Mata Rural"
    python tools/download_colingpb.py --max-informants 20 --out colingpb

The diarization model is gated: accept its terms on HuggingFace, then either log in
once with `huggingface-cli login` or pass a token through the standard HF_TOKEN
environment variable (never hardcode one here).
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import UA, human, info, resolve_out, warn  # noqa: E402

REPO = 'https://repositorio.ufpb.br'
COMMUNITY = '/jspui/handle/123456789/23184'   # CoLingPB - Corpus Linguístico da Paraíba
PAGE_SIZE = 100
MANIFEST = 'colingpb_urls.json'

def _hf_token():
    """HF_TOKEN / HUGGING_FACE_HUB_TOKEN, else the token saved by `huggingface-cli login`."""
    tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


HF_TOKEN = _hf_token()
SR = 16000
MIN_SEG_S = 2.0
MERGE_GAP_S = 0.3

ITEM_RE = re.compile(r'<a href="(/jspui/handle/123456789/\d+)">([^<]*)</a>')
WAV_RE = re.compile(r'href="(/jspui/bitstream/[^"]+\.wav)"', re.IGNORECASE)
# brPB01_g1bF01, and the university-educated variants brPB01_g2auF01
INFORMANT_RE = re.compile(r'^brPB\d+_g\d+[a-z]+[FM]\d+$')

warnings.filterwarnings('ignore')


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode('utf-8', 'replace')


# ── discovery ────────────────────────────────────────────────────────────────

def browse_items() -> dict[str, str]:
    """{handle path: item title} for every item in the CoLingPB community."""
    items: dict[str, str] = {}
    offset = 0
    while True:
        page = fetch(f'{REPO}{COMMUNITY}/browse?type=title'
                     f'&rpp={PAGE_SIZE}&offset={offset}')
        found = {h: html.unescape(t) for h, t in ITEM_RE.findall(page)
                 if 'CoLingPB' in html.unescape(t)}
        new = {h: t for h, t in found.items() if h not in items}
        if not new:
            break
        items.update(new)
        print(f'\r  browsing: {len(items)} items   ', end='', flush=True)
        offset += PAGE_SIZE
    print(flush=True)
    return items


def informants(items: dict[str, str]) -> list[dict]:
    """Keep the items that are recordings, parsing region/city out of the title."""
    out = []
    for handle, title in items.items():
        parts = title.split('/')
        # CoLingPB / region / city / informant — anything else is documentation
        if len(parts) != 4 or not INFORMANT_RE.match(parts[3]):
            continue
        out.append({'name': parts[3], 'region': parts[1], 'city': parts[2],
                    'handle': handle})
    return sorted(out, key=lambda d: d['name'])


def resolve_wav(entry: dict) -> dict:
    """Attach the WAV bitstream URL by reading the item page."""
    try:
        page = fetch(REPO + entry['handle'])
        m = WAV_RE.search(page)
        entry['url'] = REPO + m.group(1) if m else None
    except Exception as e:
        entry['url'] = None
        entry['error'] = str(e)
    return entry


def discover(jobs: int) -> list[dict]:
    print(f'  discovering items under {REPO}{COMMUNITY}', flush=True)
    entries = informants(browse_items())
    info(f'{len(entries)} informant items')

    print(f'  resolving WAV bitstreams ({jobs} workers)...', flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        entries = list(ex.map(resolve_wav, entries))

    missing = [e['name'] for e in entries if not e.get('url')]
    if missing:
        warn(f'{len(missing)} items have no WAV bitstream: '
             f'{", ".join(missing[:5])}{" ..." if len(missing) > 5 else ""}')
    return [e for e in entries if e.get('url')]


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    # Tolerate the older [[name, url], ...] shape.
    if data and isinstance(data[0], (list, tuple)):
        return [{'name': n, 'url': u, 'region': '', 'city': ''} for n, u in data]
    return data


# ── audio ────────────────────────────────────────────────────────────────────

def download_wav(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers=UA)
    raw = urllib.request.urlopen(req, timeout=120).read()
    audio, sr = sf.read(io.BytesIO(raw))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def merge_segments(segments, gap=MERGE_GAP_S):
    if not segments:
        return []
    out = [list(segments[0])]
    for s, e, spk in segments[1:]:
        if spk == out[-1][2] and s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e, spk])
    return [tuple(x) for x in out]


def process_one(entry: dict, out_root: Path, pipeline) -> list[dict]:
    name, url = entry['name'], entry['url']
    spk_dirs = [out_root / f'{name}_spkA', out_root / f'{name}_spkB']
    if all(d.exists() and len(list(d.glob('*.wav'))) > 0 for d in spk_dirs):
        counts = [len(list(d.glob('*.wav'))) for d in spk_dirs]
        print(f'  [skip] {name}: already done ({counts[0]}/{counts[1]} segs)', flush=True)
        return [{**entry, 'speaker_dir': d.name, 'n_segments': n}
                for d, n in zip(spk_dirs, counts)]

    print(f'  downloading {name} ...', flush=True)
    try:
        audio = download_wav(url)
    except Exception as e:
        warn(f'download failed for {name}: {e}')
        return []

    print(f'  {name}: {len(audio)/SR:.1f}s, diarizing ...', flush=True)
    try:
        import torch
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        result = pipeline({'waveform': waveform, 'sample_rate': SR})
        diarization = result.speaker_diarization
    except Exception as e:
        warn(f'diarization failed for {name}: {e}')
        return []

    seen: dict[str, int] = {}

    def spk_idx(label):
        return seen.setdefault(label, len(seen))

    segments = sorted(((turn.start, turn.end, spk_idx(spk))
                       for turn, _, spk in diarization.itertracks(yield_label=True)),
                      key=lambda x: x[0])
    segments = merge_segments(segments)

    # More than two voices detected — keep the two that talk the most
    if len({s[2] for s in segments}) > 2:
        from collections import Counter
        dur = Counter()
        for s, e, spk in segments:
            dur[spk] += e - s
        top2 = {spk for spk, _ in dur.most_common(2)}
        remap = {old: new for new, old in enumerate(sorted(top2))}
        segments = [(s, e, remap[spk]) for s, e, spk in segments if spk in top2]

    for d in spk_dirs:
        d.mkdir(parents=True, exist_ok=True)

    counters = [0, 0]
    for s, e, spk in segments:
        if spk > 1 or e - s < MIN_SEG_S:
            continue
        clip = audio[int(s * SR):int(e * SR)]
        sf.write(spk_dirs[spk] / f'{name}_{counters[spk]:04d}.wav', clip, SR,
                 subtype='PCM_16')
        counters[spk] += 1

    print(f'  {name}: spkA={counters[0]} segs, spkB={counters[1]} segs', flush=True)
    return [{**entry, 'speaker_dir': d.name, 'n_segments': n}
            for d, n in zip(spk_dirs, counters)]


def load_pipeline():
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    pipeline = PyannotePipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1', token=HF_TOKEN)
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device('cuda'))
        info('diarization on GPU')
    else:
        warn('no CUDA device — diarizing on CPU, this is much slower')
    return pipeline


def main():
    ap = argparse.ArgumentParser(
        description='Download + diarize the CoLingPB interview corpus')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/colingpb)')
    ap.add_argument('--urls', default=None,
                    help=f'manifest JSON to use (default: {MANIFEST} under --out)')
    ap.add_argument('--refresh', action='store_true',
                    help='re-discover the manifest even if one is cached')
    ap.add_argument('--discover-only', action='store_true',
                    help='write the manifest and exit (no audio, no HF_TOKEN needed)')
    ap.add_argument('--regions', default=None,
                    help='comma-separated regions to keep, e.g. "Mata Urbana,Sertão Rural"')
    ap.add_argument('--max-informants', type=int, default=None)
    ap.add_argument('--min_wavs', type=int, default=3,
                    help='report speaker dirs with fewer than N segments')
    ap.add_argument('--start', type=int, default=0, help='resume at this index')
    ap.add_argument('--jobs', type=int, default=6,
                    help='parallel workers for manifest discovery (default: 6)')
    args = ap.parse_args()

    out_root = resolve_out(args.out, 'colingpb')
    manifest_path = Path(args.urls) if args.urls else out_root / MANIFEST
    print('=== CoLingPB ===')
    print(f'  output: {out_root}')

    if manifest_path.exists() and not args.refresh:
        entries = load_manifest(manifest_path)
        info(f'{len(entries)} entries from {manifest_path} (--refresh to rebuild)')
    else:
        entries = discover(args.jobs)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=1),
                                 encoding='utf-8')
        info(f'manifest -> {manifest_path}')

    if args.regions:
        wanted = {r.strip() for r in args.regions.split(',') if r.strip()}
        available = sorted({e.get('region', '') for e in entries} - {''})
        unknown = wanted - set(available)
        if unknown:
            warn(f'unknown regions: {sorted(unknown)}')
            warn(f'available: {available}')
            return 1
        entries = [e for e in entries if e.get('region') in wanted]
        info(f'{len(entries)} informants in {sorted(wanted)}')

    if args.max_informants:
        entries = entries[:args.max_informants]

    if args.discover_only:
        for e in entries[:10]:
            print(f'  {e["name"]:22s} {e.get("region",""):18s} {e.get("city","")}')
        if len(entries) > 10:
            print(f'  ... and {len(entries) - 10} more')
        return 0

    if not HF_TOKEN:
        raise SystemExit(
            'Log in with `huggingface-cli login` (or set HF_TOKEN) using a HuggingFace token with access to '
            'pyannote/speaker-diarization-3.1. '
            'Use --discover-only to build the manifest without it.')

    print('  loading pyannote pipeline ...', flush=True)
    pipeline = load_pipeline()

    print(f'  processing {len(entries)} informants ...', flush=True)
    rows = []
    for i, entry in enumerate(entries):
        if i < args.start:
            continue
        print(f'[{i+1}/{len(entries)}] {entry["name"]}', flush=True)
        rows.extend(process_one(entry, out_root, pipeline))

    if rows:
        meta = out_root / 'metadata.csv'
        with open(meta, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['speaker_dir', 'name', 'region', 'city',
                                              'n_segments', 'handle'],
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r['speaker_dir']))
        info(f'metadata.csv ({len(rows)} speaker dirs) -> {meta}')

    spk_dirs = [d for d in out_root.iterdir() if d.is_dir()]
    kept = [d for d in spk_dirs if len(list(d.glob('*.wav'))) >= args.min_wavs]
    size = sum(f.stat().st_size for f in out_root.rglob('*.wav'))
    info(f'{len(kept)}/{len(spk_dirs)} speaker dirs with >= {args.min_wavs} segments')
    info(f'{human(size)} in {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
