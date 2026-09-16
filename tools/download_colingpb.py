"""
Download CoLingPB interview WAVs and diarize into 2-speaker segments using pyannote.

Pipeline per recording:
  1. Download WAV from repositorio.ufpb.br
  2. Resample to 16 kHz mono, save temp file
  3. pyannote/speaker-diarization-3.1 → speaker timeline
  4. Merge short segments, discard < MIN_SEG_S
  5. Save segments as colingpb/{informant}_spkA/{informant}_{seg:04d}.wav
                       colingpb/{informant}_spkB/{informant}_{seg:04d}.wav

Usage:
    export HF_TOKEN=hf_...          # needs access to pyannote/speaker-diarization-3.1
    python download_colingpb.py
    python download_colingpb.py --jobs 1 --out colingpb

Requires a HuggingFace token with access to the gated pyannote diarization model,
passed via the standard HF_TOKEN environment variable (never hardcode tokens here).
"""
import argparse
import json
import os
import urllib.request
import io
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

HF_TOKEN    = os.environ.get('HF_TOKEN')
URLS_JSON   = '/tmp/colingpb_urls.json'
SR          = 16000
MIN_SEG_S   = 2.0
MERGE_GAP_S = 0.3

warnings.filterwarnings('ignore')


def download_wav(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=60).read()
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


def process_one(name: str, url: str, out_root: Path, pipeline):
    spk_dirs = [out_root / f'{name}_spkA', out_root / f'{name}_spkB']
    if all(d.exists() and len(list(d.glob('*.wav'))) > 0 for d in spk_dirs):
        n0 = len(list(spk_dirs[0].glob('*.wav')))
        n1 = len(list(spk_dirs[1].glob('*.wav')))
        print(f'  [skip] {name}: already done ({n0}/{n1} segs)', flush=True)
        return

    print(f'  Downloading {name} ...', flush=True)
    try:
        audio = download_wav(url)
    except Exception as e:
        print(f'  ! download failed: {e}')
        return

    print(f'  {name}: {len(audio)/SR:.1f}s  diarizing ...', flush=True)
    try:
        import torch
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        result = pipeline({'waveform': waveform, 'sample_rate': SR})
        diarization = result.speaker_diarization
    except Exception as e:
        print(f'  ! diarization failed: {e}')
        return

    # Collect segments per speaker label
    raw = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw.append((turn.start, turn.end, speaker))

    # Map arbitrary speaker labels to 0/1
    seen = {}
    def spk_idx(label):
        if label not in seen:
            seen[label] = len(seen)
        return seen[label]

    segments = [(s, e, spk_idx(spk)) for s, e, spk in raw]
    segments = sorted(segments, key=lambda x: x[0])
    segments = merge_segments(segments)

    # If more than 2 speakers detected, keep only the 2 most active
    if len(set(s[2] for s in segments)) > 2:
        from collections import Counter
        dur_by_spk = Counter()
        for s, e, spk in segments:
            dur_by_spk[spk] += e - s
        top2 = {spk for spk, _ in dur_by_spk.most_common(2)}
        segments = [(s, e, spk) for s, e, spk in segments if spk in top2]
        # Remap to 0/1
        remap = {old: new for new, old in enumerate(sorted(top2))}
        segments = [(s, e, remap[spk]) for s, e, spk in segments]

    for d in spk_dirs:
        d.mkdir(parents=True, exist_ok=True)

    counters = [0, 0]
    for s, e, spk in segments:
        if spk > 1:
            continue
        dur = e - s
        if dur < MIN_SEG_S:
            continue
        lo = int(s * SR)
        hi = int(e * SR)
        clip = audio[lo:hi]
        fname = f'{name}_{counters[spk]:04d}.wav'
        sf.write(spk_dirs[spk] / fname, clip, SR, subtype='PCM_16')
        counters[spk] += 1

    print(f'  {name}: spkA={counters[0]} segs, spkB={counters[1]} segs', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',      default='colingpb')
    ap.add_argument('--urls',     default=URLS_JSON)
    ap.add_argument('--min_wavs', type=int, default=3)
    ap.add_argument('--start',    type=int, default=0, help='start at index')
    args = ap.parse_args()

    if not HF_TOKEN:
        raise SystemExit('Set HF_TOKEN to a HuggingFace token with access to '
                          'pyannote/speaker-diarization-3.1 (export HF_TOKEN=hf_...)')

    entries = json.load(open(args.urls))
    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)

    print('Loading pyannote pipeline ...', flush=True)
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    pipeline = PyannotePipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1',
        token=HF_TOKEN,
    )
    pipeline = pipeline.to(torch.device('cuda'))
    print('Pipeline ready.', flush=True)

    print(f'Processing {len(entries)} informants ...', flush=True)
    for i, (name, url) in enumerate(entries):
        if i < args.start:
            continue
        print(f'[{i+1}/{len(entries)}] {name}', flush=True)
        process_one(name, url, out_root, pipeline)

    spk_dirs = [d for d in out_root.iterdir() if d.is_dir()]
    kept = [d for d in spk_dirs if len(list(d.glob('*.wav'))) >= args.min_wavs]
    print(f'\nDone. {len(kept)}/{len(spk_dirs)} speaker-dirs with >= {args.min_wavs} segments.')
    print(f'All data under {out_root}/')


if __name__ == '__main__':
    main()
