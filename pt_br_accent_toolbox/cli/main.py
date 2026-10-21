from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog='pt-br-accent-toolbox',
        description='Feature extraction and analysis tools for PT-BR speech.',
    )
    sub = parser.add_subparsers(dest='command', help='Sub-command to run')

    # ── extract ────────────────────────────────────────────────────────────
    pext = sub.add_parser('extract', help='Extract features for speakers')
    pext.add_argument('--speakers', required=True,
                      help='JSON file: {speaker_id: [audio_path, ...]}')
    pext.add_argument('--features', nargs='+', default=['formants', 'mfcc'],
                      help="Feature names: spec zipa px formants mfcc ssl_ecapa ssl_xlsr ssl_hubert ssl_w2vbert vowel_harmony")
    pext.add_argument('--texts', help='JSON file: {speaker_id: [transcript, ...]} (vowel_harmony only)')
    pext.add_argument('--out', required=True, help='Output directory')

    # ── markers ────────────────────────────────────────────────────────────
    pmk = sub.add_parser('markers', help='Extract phonological marker frames')
    pmk.add_argument('audio', help='Audio file path')
    pmk.add_argument('--out', help='Output JSON path (default: stdout)')

    # ── phonemes ───────────────────────────────────────────────────────────
    pph = sub.add_parser('phonemes', help='Extract phoneme sequence from audio')
    pph.add_argument('audio', help='Audio file path')
    pph.add_argument('--out', help='Output JSON path (default: stdout)')

    # ── harmony ────────────────────────────────────────────────────────────
    phm = sub.add_parser('harmony', help='Pretonic vowel-harmony descriptors for one speaker')
    phm.add_argument('audio', nargs='+', help="One speaker's audio files")
    phm.add_argument('--texts', help='Text file with one transcript per audio file, in order '
                                     '(default: greedy CTC transcription)')
    phm.add_argument('--tokens-out', help='Also write the per-token measurements to this CSV')
    phm.add_argument('--device', help='torch device (default: cuda if available)')
    phm.add_argument('--out', help='Output JSON path (default: stdout)')

    # ── annotations ────────────────────────────────────────────────────────
    pann = sub.add_parser('annotations', help='List/load speaker annotations')
    pann.add_argument('--db', help='Annotations DB path')
    pann.add_argument('--marker', choices=['s_coda', 'r_coda', 'dt_palat'])
    pann.add_argument('--value', help='Filter by annotation value')
    pann.add_argument('--rows', action='store_true',
                      help='Dump the raw per-annotator rows (keeps dataset, annotator, source)')
    pann.add_argument('--include-auto', action='store_true',
                      help="Also include the script-written rows (annotator 'auto')")
    pann.add_argument('--out', help='Output JSON path (default: stdout)')

    args = parser.parse_args()

    if args.command == 'extract':
        run_extract(args)
    elif args.command == 'markers':
        run_markers(args)
    elif args.command == 'phonemes':
        run_phonemes(args)
    elif args.command == 'harmony':
        run_harmony(args)
    elif args.command == 'annotations':
        run_annotations(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_extract(args):
    from ..api.pipeline import FeaturePipeline
    speakers = json.load(open(args.speakers))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = json.load(open(args.texts)) if args.texts else None
    pipe = FeaturePipeline()
    results = pipe.extract(speakers, args.features, texts=texts)

    for fname, spk_dict in results.items():
        data: dict[str, list] = {'speaker': [], 'vector': []}
        for spk, vec in spk_dict.items():
            data['speaker'].append(spk)
            data['vector'].append(vec.tolist())
        np.savez(str(out_dir / f'{fname}.npz'),
                 speaker=np.array(data['speaker']),
                 vector=np.array(data['vector'], np.float32))
        print(f'  saved {out_dir}/{fname}.npz: {len(spk_dict)} speakers')


def run_markers(args):
    from ..alignment.zipa import load_audio, extract_marker_frames
    audio = load_audio(args.audio)
    markers = extract_marker_frames(audio)

    out = {'n_markers': len(markers), 'markers': [
        {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
         for k, v in m.items()}
        for m in markers
    ]}

    if args.out:
        json.dump(out, open(args.out, 'w'), indent=2, ensure_ascii=False)
        print(f'  saved {args.out}')
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


def run_phonemes(args):
    from ..alignment.zipa import load_audio, extract_phoneme_sequence
    audio = load_audio(args.audio)
    phones = extract_phoneme_sequence(audio)

    out = {'phonemes': phones, 'n_tokens': len(phones), 'sequence': ' '.join(phones)}

    if args.out:
        json.dump(out, open(args.out, 'w'), indent=2, ensure_ascii=False)
        print(f'  saved {args.out}')
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


def run_harmony(args):
    import csv
    from ..features.vowel_harmony import (HarmonyMeasurer, load_reference_cohort,
                                          speaker_descriptors)
    texts = None
    if args.texts:
        texts = Path(args.texts).read_text(encoding='utf-8').splitlines()
        if len(texts) != len(args.audio):
            sys.exit(f'--texts has {len(texts)} lines for {len(args.audio)} audio files')
    hm = HarmonyMeasurer(device=args.device)
    rows = hm.speaker_tokens(args.audio, texts)
    desc = speaker_descriptors(rows)

    # where this speaker's OI falls in the Exp 56 regional distributions
    ref = load_reference_cohort(gate_ok_only=True)
    pct = {}
    if np.isfinite(desc['OI']):
        for macro in ('NE', 'SE'):
            oi = np.array([r['OI'] for r in ref if r['macro'] == macro])
            pct[macro] = round(float(np.mean(oi <= desc['OI']) * 100), 1)

    def _num(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), 4)
        return v

    out = {'n_files': len(args.audio), 'n_tokens': len(rows), 'gate_ok': int(desc['gate_ok']),
           'OI_percentile_in_reference': pct,
           'descriptors': {k: _num(v) for k, v in desc.items()}}
    if not desc['gate_ok']:
        out['warning'] = ('speaker fails the anchor or token gate: the descriptors '
                          'should not be trusted (more / longer audio usually fixes it)')

    if args.tokens_out and rows:
        keys = list(rows[0].keys())
        with open(args.tokens_out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'  saved {args.tokens_out}: {len(rows)} tokens', file=sys.stderr)

    if args.out:
        json.dump(out, open(args.out, 'w'), indent=2, ensure_ascii=False)
        print(f'  saved {args.out}')
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


def run_annotations(args):
    from ..data.annotations import (get_annotated_speakers, load_annotation_rows,
                                    load_annotations)

    if args.rows:
        rows = load_annotation_rows()
        if not args.include_auto:
            rows = [r for r in rows if r.get('source') != 'auto']
        if args.marker:
            rows = [r for r in rows
                    if r.get(args.marker) and
                    (args.value is None or r[args.marker] == args.value)]
        out = rows
    elif args.marker:
        spks = get_annotated_speakers(args.db, args.marker, args.value,
                                      include_auto=args.include_auto)
        out = {'speakers': spks, 'n': len(spks)}
    else:
        out = load_annotations(args.db, include_auto=args.include_auto)

    if args.out:
        json.dump(out, open(args.out, 'w'), indent=2, ensure_ascii=False)
        print(f'  saved {args.out}')
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))
