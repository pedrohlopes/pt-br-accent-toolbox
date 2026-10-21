"""
Download Fake-BrAccent from Kaggle and index it.

Two Kaggle releases of the same voice-conversion corpus derived from BRAccent
(five accents x two genders, every FAKE clip paired with the REAL source):

  original  erickmiguelsantos/fake-braccent   ~183 MB, 746 files
            373 REAL (.wav) + 373 FAKE (mostly .mp3, some .aac/.wav)
  d         katarinaveljovic/fake-braccent-d  ~400 MB zip, 1546 files
            the same 373 FAKE clips; REAL extended to 1172 .wav (more BRAccent
            sentences per speaker, unprefixed file names) + a .docx note

The label is taken from the FAKE/REAL folder, never from the file name.

Layout on Kaggle (both): <root>/{Accent}/{Masculino,Feminino}/{FAKE,REAL}/*.
File names carry the BRAccent metadata:

  [FAKE_|REAL_]converted_{spk}_frase_{n}_{date}_{age}_{city}_{state}_{gender}_{education}.ext

(a few speakers carry an extra number after {n} and/or omit {age}; both parse).

Output ({ACCENTS_BASE}/fake_braccent by default):

  fake_braccent/
    original/Fake_BrAccent/{Accent}/{Gender}/{FAKE,REAL}/...   as shipped
    d/FakeBrAccent 2/{Accent}/{Gender}/{FAKE,REAL}/...
    metadata.csv      one row per audio file, both variants, parsed fields
    wav16k/...        (--to-wav16k) mirror tree, 16 kHz mono PCM16 .wav

Requirements:
  pip install kaggle            + ~/.kaggle/kaggle.json (chmod 600)
  --to-wav16k needs librosa; the .mp3/.aac FAKE clips need ffmpeg on PATH

Usage:
  python download_fakebraccent.py                       # both variants
  python download_fakebraccent.py --variant d           # the extended release
  python download_fakebraccent.py --to-wav16k
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import info, resolve_out, warn  # noqa: E402

VARIANTS = {
    'original': 'erickmiguelsantos/fake-braccent',
    'd': 'katarinaveljovic/fake-braccent-d',
}
AUDIO_EXT = ('.wav', '.mp3', '.aac', '.flac', '.m4a')
ACCENT_MACRO = {'Nordestino': 'NE', 'Baiano': 'NE', 'Carioca': 'SE',
                'Fluminense': 'SE', 'Sulista': 'S'}

# converted_4rqv6rrro_frase_1_2018-08-29_60_Salvador_Bahia_Feminino_Mestrado
# converted_9er8q86jp_frase_2_36_2018-08-12_salvador_Bahia_Feminino_...   (irregular)
NAME_RE = re.compile(
    r'^(?:(?P<prefix>FAKE|REAL)_)?converted_(?P<speaker>[0-9a-z]+)_frase_(?P<sentence>\d+)_'
    r'(?:\d+_)?(?P<date>\d{4}-\d{2}-\d{2})_(?:(?P<age>\d+)_)?(?P<city>[^_]+)_(?P<state>[^_]+)_'
    r'(?P<gender_f>Masculino|Feminino)_(?P<education>[^_.\s(]+)', re.IGNORECASE)


def kaggle_download(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zips = sorted(dest.glob('*.zip'))
    if zips:
        info(f'reusing {zips[0].name}')
        return zips[0]
    print(f'  kaggle datasets download -d {slug} -p {dest}', flush=True)
    subprocess.run(['kaggle', 'datasets', 'download', '-d', slug, '-p', str(dest)],
                   check=True)
    zips = sorted(dest.glob('*.zip'))
    if not zips:
        raise FileNotFoundError(f'no zip in {dest} after kaggle download')
    return zips[0]


def parse_entry(rel: Path, variant: str) -> dict | None:
    """Kaggle-relative path -> metadata row (None for non-audio files)."""
    name = rel.name
    low = name.lower()
    if not low.endswith(AUDIO_EXT):
        # one clip in the release has no extension at all; keep it if it sits
        # in a FAKE/REAL folder
        if rel.parent.name.upper() not in ('FAKE', 'REAL') or '.' in name[-5:]:
            return None
    parts = rel.parts
    label = rel.parent.name.upper()
    row = {'variant': variant, 'path': str(rel),
           'label': {'FAKE': 'fake', 'REAL': 'real'}.get(label, ''),
           'accent': parts[-4] if len(parts) >= 4 else '',
           'gender': parts[-3] if len(parts) >= 3 else '',
           'speaker': '', 'sentence': '', 'date': '', 'age': '', 'city': '',
           'state': '', 'education': ''}
    row['macro'] = ACCENT_MACRO.get(row['accent'], '')
    m = NAME_RE.match(name)
    if m:
        row.update({k: m.group(k) or '' for k in
                    ('speaker', 'sentence', 'date', 'age', 'city', 'state', 'education')})
    return row


def extract(zip_path: Path, out: Path) -> list[Path]:
    """Extract every member (idempotent); returns paths relative to `out`."""
    rels = []
    with zipfile.ZipFile(zip_path) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            target = out / zi.filename
            rels.append(Path(zi.filename))
            if target.exists() and target.stat().st_size == zi.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    return rels


def to_wav16k(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    try:
        import librosa
        import soundfile as sf
        a, _ = librosa.load(str(src), sr=16000, mono=True)
    except Exception as e:  # mp3/aac without ffmpeg, or a broken clip
        warn(f'cannot decode {src.name}: {type(e).__name__}')
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), a, 16000, subtype='PCM_16')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/fake_braccent)')
    ap.add_argument('--variant', choices=['original', 'd', 'both'], default='both')
    ap.add_argument('--to-wav16k', action='store_true',
                    help='also write a 16 kHz mono .wav mirror under wav16k/')
    ap.add_argument('--keep-zip', action='store_true')
    args = ap.parse_args()

    out = resolve_out(args.out, 'fake_braccent')
    variants = list(VARIANTS) if args.variant == 'both' else [args.variant]
    rows = []
    for v in variants:
        slug = VARIANTS[v]
        print(f'\n=== {v} ({slug}) ===', flush=True)
        vdir = out / v
        tmp = out / f'_download_{v}'
        zip_path = kaggle_download(slug, tmp)
        rels = extract(zip_path, vdir)
        info(f'{len(rels)} files under {vdir}')
        if not args.keep_zip:
            shutil.rmtree(tmp, ignore_errors=True)
        for rel in rels:
            r = parse_entry(rel, v)
            if r is None:
                continue
            r['path'] = f'{v}/{rel}'
            if args.to_wav16k:
                dst = (out / 'wav16k' / v / rel).with_suffix('.wav')
                r['wav16k'] = str(dst.relative_to(out)) if to_wav16k(vdir / rel, dst) else ''
            rows.append(r)

    if rows:
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / 'metadata.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        n_fake = sum(r['label'] == 'fake' for r in rows)
        n_real = sum(r['label'] == 'real' for r in rows)
        n_parsed = sum(bool(r['speaker']) for r in rows)
        info(f'metadata.csv: {len(rows)} clips ({n_real} real, {n_fake} fake), '
             f'{n_parsed} with parsed speaker metadata')
    print(f'\nDone -> {out}')


if __name__ == '__main__':
    main()
