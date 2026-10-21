"""
Download the Blizzard Challenge 2027 training material and (optionally) export
NURC-TTS as per-speaker 16 kHz wavs with transcripts.

Blizzard 2027 is "Accented Zero-Shot TTS in Spontaneous Portuguese Speech".
The data description is on Zenodo (record 22208787, CC-BY-4.0):

    https://zenodo.org/records/22208787
    https://nilc-nlp.github.io/ssw-2027/blizzard-challenge-rules.html

and the audio is on three public, ungated HuggingFace datasets:

    nurc_tts_24khz  nilc-nlp/nurc_tts_24khz   ~82 GB  24 kHz   splits sao_paulo, recife
    nurc_tts        nilc-nlp/nurc_tts        ~150 GB  44.1 kHz same segments
    falar_tts       inesc-id/FalAR-TTS        ~55 GB  24 kHz   European Portuguese parliament

NURC-TTS is spontaneous speech (interviews DID, dialogues D2, lectures EF)
re-segmented to 3-30 s clips with human transcripts: 231 h from São Paulo
(NURC-SP) and 241 h from Recife (NURC-RE). Since the two splits are the two
sides of the Northeast/Southeast isoglosses, it is a natural accent corpus.

Output ({ACCENTS_BASE}/blizzard2027 by default; same layout as the accents
project's copy):

    blizzard2027/
      nurc_tts_24khz/data/{recife,sao_paulo}-*.parquet   as shipped
      nurc_tts_44k/data/...
      falar_tts/data/train-*.parquet
      nurc_wav16k/                                       (--export-wav)
        {split}/{inquiry}__{speaker}/{segment}.wav       16 kHz mono PCM16
        metadata.csv                                     one row per clip, with text

Things worth knowing before you use NURC-TTS for accent work:

  * Recife clips are DUPLICATED across two TextGrid tiers: every segment
    appears once as "<Spk>-ue" and once as "<Spk>-nurc", same start/end
    times, near-identical text (the -nurc tier keeps prosodic marks such as
    "..."). The export keeps one tier (--recife-tier, default ue).
  * The `speaker` field is per inquiry, not global ("Inf1-ue", "SPEAKER 2"),
    and comes from WhisperX diarization on the SP side, which the corpus
    authors note may be slightly wrong. Treat {inquiry}__{speaker} as the
    speaker key.
  * Documenters ("Doc-*", Recife) are interviewers, not the informants the
    corpus was built around; they are excluded from the export unless
    --include-documenters. SP has no such tag (its documenter is just another
    "SPEAKER n", usually the one with few clips — use --min-clips).
  * The audio comes from analog reel-to-reel tapes (1970s-90s): the sampling
    rate overstates the bandwidth.

Usage:
    python tools/download_blizzard2027.py                          # nurc_tts_24khz, both splits
    python tools/download_blizzard2027.py --max-shards 2 --export-wav
    python tools/download_blizzard2027.py --splits recife --export-wav --max-per-speaker 200
    python tools/download_blizzard2027.py --datasets all           # ~290 GB
    python tools/download_blizzard2027.py --export-wav --skip-download   # re-export local parquets

Needs `huggingface_hub`, `pyarrow`, and for --export-wav `soundfile` + `librosa`.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import human, info, resolve_out, warn  # noqa: E402

# HF's free-tier rate limit (1000 req / 5 min) breaks a plain snapshot_download
# of hundreds of shards; xet off + 2 workers + an outer retry loop got the
# accents project's copy through.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

DATASETS = {
    'nurc_tts_24khz': {'repo': 'nilc-nlp/nurc_tts_24khz', 'dir': 'nurc_tts_24khz',
                       'splits': ('sao_paulo', 'recife')},
    'nurc_tts':       {'repo': 'nilc-nlp/nurc_tts', 'dir': 'nurc_tts_44k',
                       'splits': ('sao_paulo', 'recife')},
    'falar_tts':      {'repo': 'inesc-id/FalAR-TTS', 'dir': 'falar_tts',
                       'splits': ('train',)},
}
NURC_REGION = {'recife': 'PE', 'sao_paulo': 'SP'}
NURC_MACRO = {'recife': 'NE', 'sao_paulo': 'SE'}
TIER_RE = re.compile(r'-(ue|nurc)$')
SEG_TIMES_RE = re.compile(r'_([\d.]+)_([\d.]+)\.wav$')
META_COLS = ['inquiry', 'segment', 'speaker', 'duration', 'text', 'inq_type',
             'inq_recording_year', 'inq_gender', 'inq_quality', 'inq_themes',
             'inq_age_group']
MAX_ATTEMPTS = 50


def _token():
    tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


# ── download ─────────────────────────────────────────────────────────────────

def shard_patterns(repo: str, splits, max_shards: int | None, token) -> list[str]:
    """allow_patterns for snapshot_download: whole splits, or their first N shards."""
    if not max_shards:
        return ['README.md'] + [f'data/{s}-*' for s in splits]
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type='dataset', token=token)
    keep = ['README.md']
    for s in splits:
        keep += sorted(f for f in files if f.startswith(f'data/{s}-'))[:max_shards]
    return keep


def fetch(name: str, out: Path, splits, max_shards: int | None, token) -> bool:
    from huggingface_hub import snapshot_download
    ds = DATASETS[name]
    target = out / ds['dir']
    patterns = shard_patterns(ds['repo'], splits, max_shards, token)
    print(f'\n=== {name} ({ds["repo"]}) -> {target}', flush=True)
    print(f'  patterns: {patterns if len(patterns) < 8 else patterns[:4] + ["..."]}', flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            snapshot_download(repo_id=ds['repo'], repo_type='dataset', local_dir=str(target),
                              allow_patterns=patterns, token=token, max_workers=2)
            n = len(list((target / 'data').glob('*.parquet')))
            size = sum(f.stat().st_size for f in (target / 'data').glob('*.parquet'))
            info(f'{name}: {n} shards, {human(size)}')
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:                       # rate limits, resets: back off and resume
            wait = min(30 * attempt, 300)
            warn(f'attempt {attempt} failed ({type(e).__name__}: {str(e)[:160]}); retry in {wait}s')
            time.sleep(wait)
    warn(f'{name}: gave up after {MAX_ATTEMPTS} attempts (re-run to resume)')
    return False


# ── NURC export ──────────────────────────────────────────────────────────────

def _local_shards(root: Path, split: str, max_shards: int | None) -> list[Path]:
    shards = sorted((root / 'data').glob(f'{split}-*.parquet'))
    return shards[:max_shards] if max_shards else shards


def plan_tiers(shards: list[Path], prefer: str) -> dict[str, str | None]:
    """Recife: which tier suffix to keep per inquiry (metadata columns only).

    An inquiry carrying both tiers keeps `prefer`; one with a single tier keeps
    it; None means the inquiry has no tier suffix at all (keep everything).
    """
    import pyarrow.parquet as pq
    tiers: dict[str, set] = defaultdict(set)
    for f in shards:
        t = pq.read_table(f, columns=['inquiry', 'speaker'])
        for inq, spk in zip(t['inquiry'].to_pylist(), t['speaker'].to_pylist()):
            m = TIER_RE.search(spk or '')
            tiers[inq].add(m.group(1) if m else None)
    plan = {}
    for inq, ts in tiers.items():
        ts.discard(None)
        plan[inq] = (prefer if prefer in ts else next(iter(ts))) if ts else None
    return plan


def _resample_wav(raw: bytes, sr_out: int):
    import librosa
    import numpy as np
    import soundfile as sf
    a, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    a = a.mean(axis=1)
    if sr != sr_out:
        a = librosa.resample(a, orig_sr=sr, target_sr=sr_out)
    return np.clip(a, -1.0, 1.0)


def export_nurc(root: Path, out: Path, splits, args) -> int:
    """Parquet shards -> {split}/{inquiry}__{speaker}/{segment}.wav + metadata.csv."""
    import pyarrow.parquet as pq
    import soundfile as sf

    wav_root = out / 'nurc_wav16k'
    meta_path = wav_root / 'metadata.csv'
    keys = ['split', 'region', 'macro', 'speaker_key', 'role', 'path'] + META_COLS
    have = set()
    if meta_path.exists():
        with open(meta_path, newline='', encoding='utf-8') as fh:
            have = {r['path'] for r in csv.DictReader(fh)}
    fh = open(meta_path, 'a', newline='', encoding='utf-8') if have else None
    if fh is None:
        wav_root.mkdir(parents=True, exist_ok=True)
        fh = open(meta_path, 'w', newline='', encoding='utf-8')
        csv.DictWriter(fh, fieldnames=keys).writeheader()
    w = csv.DictWriter(fh, fieldnames=keys)

    per_spk: Counter = Counter()
    stats: Counter = Counter()
    try:
        for split in splits:
            shards = _local_shards(root, split, args.max_shards)
            if not shards:
                warn(f'no local {split} shards under {root}/data — download first')
                continue
            plan = plan_tiers(shards, args.recife_tier) if split == 'recife' else {}
            print(f'  exporting {split}: {len(shards)} shard(s)', flush=True)
            for f in shards:
                pf = pq.ParquetFile(f)
                for rg in range(pf.metadata.num_row_groups):
                    batch = pf.read_row_group(rg, columns=META_COLS + ['audio']).to_pylist()
                    for r in batch:
                        spk = r['speaker'] or 'unknown'
                        m = TIER_RE.search(spk)
                        if split == 'recife' and m and plan.get(r['inquiry']) not in (None, m.group(1)):
                            stats['dup_tier'] += 1
                            continue
                        base = TIER_RE.sub('', spk)
                        role = 'documenter' if base.lower().startswith('doc') else 'speaker'
                        if role == 'documenter' and not args.include_documenters:
                            stats['documenter'] += 1
                            continue
                        key = f'{r["inquiry"]}__{base.replace(" ", "")}'
                        if args.max_per_speaker and per_spk[key] >= args.max_per_speaker:
                            stats['cap'] += 1
                            continue
                        per_spk[key] += 1
                        rel = f'{split}/{key}/{Path(r["segment"]).name}'
                        if rel in have:
                            stats['present'] += 1
                            continue
                        dst = wav_root / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            sf.write(str(dst), _resample_wav(r['audio']['bytes'], args.export_sr),
                                     args.export_sr, subtype='PCM_16')
                        except Exception as e:
                            warn(f'{rel}: {type(e).__name__}: {e}')
                            continue
                        row = {k: r.get(k) for k in META_COLS}
                        row.update(split=split, region=NURC_REGION[split], macro=NURC_MACRO[split],
                                   speaker_key=key, role=role, path=rel)
                        w.writerow(row)
                        have.add(rel)
                        stats['written'] += 1
                print(f'    {f.name}: {stats["written"]} written so far', flush=True)
    finally:
        fh.close()

    n_spk = len({p.split('/')[1] for p in have})
    info(f'{stats["written"]} clips written, {stats["present"]} already present; '
         f'skipped {stats["dup_tier"]} duplicate-tier, {stats["documenter"]} documenter, '
         f'{stats["cap"]} over --max-per-speaker')
    info(f'{len(have)} clips / {n_spk} speakers in {wav_root}')
    if args.min_clips:
        small = [s for s, n in per_spk.items() if n < args.min_clips]
        if small:
            warn(f'{len(small)} speakers have < {args.min_clips} clips (likely documenters or '
                 f'diarization fragments); filter them with metadata.csv speaker_key')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Download the Blizzard Challenge 2027 training data')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/blizzard2027)')
    ap.add_argument('--datasets', default='nurc_tts_24khz',
                    help=f'comma-separated subset of {",".join(DATASETS)}, or "all" '
                         '(default: nurc_tts_24khz)')
    ap.add_argument('--splits', default='sao_paulo,recife',
                    help='NURC-TTS splits to fetch/export (default: both)')
    ap.add_argument('--max-shards', type=int, default=None,
                    help='only the first N parquet shards per split (each ~0.5-0.8 GB)')
    ap.add_argument('--skip-download', action='store_true',
                    help='use the parquets already on disk')
    ap.add_argument('--export-wav', action='store_true',
                    help='write NURC-TTS as per-speaker wavs + metadata.csv under nurc_wav16k/')
    ap.add_argument('--export-from', default=None, choices=['nurc_tts_24khz', 'nurc_tts'],
                    help='which NURC copy to export from (default: the first one downloaded)')
    ap.add_argument('--export-sr', type=int, default=16000)
    ap.add_argument('--recife-tier', choices=['ue', 'nurc'], default='ue',
                    help='which duplicated Recife TextGrid tier to keep (default: ue)')
    ap.add_argument('--include-documenters', action='store_true',
                    help='also export the Recife documenters (interviewers)')
    ap.add_argument('--max-per-speaker', type=int, default=None,
                    help='export at most N clips per speaker')
    ap.add_argument('--min-clips', type=int, default=20,
                    help='warn about speakers with fewer exported clips (default: 20)')
    args = ap.parse_args()

    names = list(DATASETS) if args.datasets == 'all' else [s.strip() for s in args.datasets.split(',')]
    bad = [n for n in names if n not in DATASETS]
    if bad:
        raise SystemExit(f'unknown dataset(s) {bad}; choose from {list(DATASETS)}')
    nurc_splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    for s in nurc_splits:
        if s not in NURC_REGION:
            raise SystemExit(f'unknown NURC split {s!r}; choose from {list(NURC_REGION)}')

    out = resolve_out(args.out, 'blizzard2027')
    token = _token()
    print('=== Blizzard Challenge 2027 ===')
    print(f'  output: {out}')

    ok = True
    if not args.skip_download:
        for n in names:
            splits = nurc_splits if n.startswith('nurc') else DATASETS[n]['splits']
            ok = fetch(n, out, splits, args.max_shards, token) and ok

    if args.export_wav:
        src = args.export_from or next((n for n in names if n.startswith('nurc')), None)
        if src is None:
            warn('--export-wav needs a NURC dataset (nurc_tts_24khz or nurc_tts)')
            return 1
        print(f'\n=== export {src} -> nurc_wav16k ===', flush=True)
        export_nurc(out / DATASETS[src]['dir'], out, nurc_splits, args)

    print(f'\nDone -> {out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
