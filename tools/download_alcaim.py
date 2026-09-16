"""
Download and filter the Alcaim / CETUC corpus (and its sibling SMT-UFRJ corpora).

Source: https://igormq.github.io/datasets/ — Igor Quintanilha's mirror of the
PT-BR ASR corpora. Alcaim (a.k.a. CETUC) is ~145 h of read speech: 100 speakers
(50 M / 50 F) each reading the same 1000 phonetically balanced sentences.

    alcaim/{Speaker}_{G}{NNN}/{G}{NNN}-{IIII}.wav     16 kHz mono
                              {G}{NNN}-{IIII}.txt     lowercase, unpunctuated

Utterance index IIII is 0-based into the 1000-sentence list, which is the same
list published as the THLS dataset:
https://gitlab.com/lfelipesv/1000-sentences-thls-dataset (sentences.txt, latin-1).
This script fetches it, re-encodes it to UTF-8 and writes it next to the audio.

The full archive is 12.8 GB, so by default the tarball is *streamed* and only the
members that survive the filters are written to disk — you never store the 12.8 GB.
Speakers are laid out as contiguous blocks, so --max-speakers stops the stream as
soon as its quota is full. Use --keep-archive when you want the raw tarball
(resumable, re-filterable offline) instead.

Usage:
    # everything (12.8 GB on the wire, ~16 GB on disk)
    python tools/download_alcaim.py

    # the phonetically balanced 50-sentence subset used across this project
    python tools/download_alcaim.py --sentences balanced50

    # 20 speakers, 10 per gender, first 50 sentences
    python tools/download_alcaim.py --max-speakers 20 --balance-gender --sentences 1-50

    # named speakers only
    python tools/download_alcaim.py --speakers Alcione_F018,Aislam_M001

    # sibling corpora from the same mirror
    python tools/download_alcaim.py --corpus lapsbm
    python tools/download_alcaim.py --corpus sid
    python tools/download_alcaim.py --corpus voxforge

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (UA, fetch_text, http_download, human, info,  # noqa: E402
                     resolve_out, warn)

import urllib.request  # noqa: E402

MIRROR = 'https://www02.smt.ufrj.br/~igor.quintanilha'

CORPORA = {
    'alcaim': {
        'urls': [f'{MIRROR}/alcaim.tar.gz'],
        'out': 'alcaim',
        'desc': 'Alcaim / CETUC — 145 h, 100 speakers x 1000 sentences',
    },
    'lapsbm': {
        'urls': [f'{MIRROR}/lapsbm-val.tar.gz', f'{MIRROR}/lapsbm-test.tar.gz'],
        'out': 'lapsbm',
        'desc': 'LapsBM (FalaBrasil/UFPA) — 35 speakers x 20 utterances',
    },
    'sid': {
        'urls': [f'{MIRROR}/sid.tar.gz'],
        'out': 'sid',
        'desc': 'Sidney — 72 speakers, 5777 utterances',
    },
    'voxforge': {
        'urls': [f'{MIRROR}/voxforge-ptbr.tar.gz'],
        'out': 'voxforge_ptbr',
        'desc': 'VoxForge pt-BR — 111+ speakers, 4130 utterances',
    },
}

SENTENCES_URL = ('https://gitlab.com/lfelipesv/1000-sentences-thls-dataset'
                 '/-/raw/master/sentences.txt')
N_SENTENCES = 1000

# alcaim/Alcione_F018/F018-0549.wav  ->  ('Alcione_F018', 'F', '018', '0549', 'wav')
MEMBER_RE = re.compile(r'([^/]+_([MF])(\d{3}))/\2\3-(\d{4})\.(wav|txt)$', re.IGNORECASE)


class Done(Exception):
    """Raised to abort the tar stream once every quota is satisfied."""


# ── the 1000-sentence list ───────────────────────────────────────────────────

def load_sentences(cache: Path | None = None) -> list[str]:
    """Fetch the THLS/Alcaim sentence list, decoding latin-1 -> str."""
    if cache and cache.exists():
        return [l.strip() for l in cache.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f'  fetching sentence list from {SENTENCES_URL}', flush=True)
    raw = fetch_text(SENTENCES_URL, encoding='latin-1')
    sentences = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(sentences) != N_SENTENCES:
        warn(f'expected {N_SENTENCES} sentences, got {len(sentences)}')
    if cache:
        cache.write_text('\n'.join(sentences) + '\n', encoding='utf-8')
        info(f'sentences.txt (UTF-8) -> {cache}')
    return sentences


DIGRAPHS = {'lh', 'nh', 'ch', 'rr', 'ss', 'qu', 'gu'}


def _units(sentence: str) -> Counter:
    """Grapheme/digraph units — the phoneme proxy used for subset selection."""
    s = re.sub(r'[^a-záéíóúãõâêôàüç]', ' ', sentence.lower())
    feats, pos = Counter(), 0
    while pos < len(s):
        if s[pos] == ' ':
            pos += 1
            continue
        bigram = s[pos:pos + 2] if pos + 1 < len(s) else ''
        if bigram in DIGRAPHS:
            feats[bigram] += 1
            pos += 2
        else:
            feats[s[pos]] += 1
            pos += 1
    return feats


def balanced_subset(sentences: list[str], k: int = 50) -> list[int]:
    """
    Greedy phonetically balanced subset: at each step add the sentence that
    minimizes the L1 distance between the subset's unit distribution and the
    full-corpus one. Deterministic — reproduces the project's 50-sentence set
    (see accents_project notes/subset_selection_procedure.md).
    """
    feats = [_units(s) for s in sentences]
    total = Counter()
    for f in feats:
        total.update(f)
    denom = sum(total.values())
    target = {u: c / denom for u, c in total.items()}

    chosen: set[int] = set()
    cur, cur_total = Counter(), 0
    for _ in range(k):
        best_d, best_i = float('inf'), None
        for i, f in enumerate(feats):
            if i in chosen:
                continue
            ft = sum(f.values())
            if ft == 0:
                continue
            merged, n = cur + f, cur_total + ft
            d = sum(abs(merged.get(u, 0) / n - target.get(u, 0))
                    for u in set(target) | set(merged))
            if d < best_d:
                best_d, best_i = d, i
        chosen.add(best_i)
        cur.update(feats[best_i])
        cur_total += sum(feats[best_i].values())
    return sorted(chosen)


def parse_sentence_spec(spec: str, sentences: list[str]) -> set[int] | None:
    """
    --sentences accepts:
      all | balanced50 | balanced:<k> | 1-50 | 1,7,9 | @path/to/file (1-based)
    Returns a set of 0-based indices, or None for "keep everything".
    """
    spec = spec.strip()
    if spec in ('all', ''):
        return None
    if spec == 'balanced50':
        return set(balanced_subset(sentences, 50))
    if spec.startswith('balanced:'):
        return set(balanced_subset(sentences, int(spec.split(':', 1)[1])))
    if spec.startswith('@'):
        spec = Path(spec[1:]).read_text(encoding='utf-8')

    keep: set[int] = set()
    for part in re.split(r'[\s,]+', spec):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            keep.update(range(int(lo) - 1, int(hi)))
        else:
            keep.add(int(part) - 1)
    return keep


def parse_speakers(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    if spec.startswith('@'):
        spec = Path(spec[1:]).read_text(encoding='utf-8')
    return {t.strip() for t in re.split(r'[\s,]+', spec) if t.strip()}


# ── the filter ───────────────────────────────────────────────────────────────

class Filter:
    """
    Decides which tar members to keep, and tracks quotas so the caller can stop
    reading the stream early. Speakers appear as contiguous blocks in these
    archives, so --max-speakers is satisfied the moment the quota is full and a
    member for an unselected speaker shows up.
    """

    def __init__(self, speakers=None, genders=None, sentence_ids=None,
                 want_txt=True, max_utts=None, max_speakers=None,
                 balance_gender=False):
        self.speakers = speakers
        self.genders = genders
        self.sentence_ids = sentence_ids
        self.want_txt = want_txt
        self.max_utts = max_utts
        self.max_speakers = max_speakers
        self.balance_gender = balance_gender

        self.taken: dict[str, str] = {}      # speaker -> gender, in archive order
        self.utts: Counter = Counter()       # speaker -> wavs kept

    def _quota_full(self) -> bool:
        if not self.max_speakers:
            return False
        if not self.balance_gender:
            return len(self.taken) >= self.max_speakers
        per = self.max_speakers // 2
        by_gender = Counter(self.taken.values())
        return by_gender['M'] >= per and by_gender['F'] >= per

    def _admit(self, spk: str, gender: str) -> bool:
        """Can this newly-seen speaker join the selection?"""
        if not self.max_speakers:
            return True
        if self.balance_gender:
            per = self.max_speakers // 2
            return Counter(self.taken.values())[gender] < per
        return len(self.taken) < self.max_speakers

    def __call__(self, name: str):
        """Return (speaker, gender, sentence_idx, ext) or None. Raises Done."""
        m = MEMBER_RE.search(name)
        if not m:
            return None
        spk, gender = m.group(1), m.group(2).upper()
        idx, ext = int(m.group(4)), m.group(5).lower()

        if self.speakers is not None and spk not in self.speakers:
            return None
        if self.genders is not None and gender not in self.genders:
            return None

        if spk not in self.taken:
            if not self._admit(spk, gender):
                if self._quota_full():
                    raise Done
                return None
            self.taken[spk] = gender

        if ext == 'txt' and not self.want_txt:
            return None
        if self.sentence_ids is not None and idx not in self.sentence_ids:
            return None
        if ext == 'wav' and self.max_utts and self.utts[spk] >= self.max_utts:
            return None
        return spk, gender, idx, ext


# ── extraction ───────────────────────────────────────────────────────────────

def iter_stream(url: str):
    """Yield (name, tarfile, member) from a remote tar.gz without storing it."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        with tarfile.open(fileobj=r, mode='r|gz') as tf:
            for member in tf:
                if member.isfile():
                    yield member.name, tf, member


def iter_archive(path: Path):
    """Yield (name, tarfile, member) from a tarball already on disk."""
    with tarfile.open(path, 'r:gz') as tf:
        for member in tf:
            if member.isfile():
                yield member.name, tf, member


def extract(members, out_root: Path, filt: Filter):
    rows, written, skipped = [], 0, 0
    try:
        for name, tf, member in members:
            hit = filt(name)
            if not hit:
                continue
            spk, gender, idx, ext = hit

            dest = out_root / spk / Path(name).name
            if dest.exists() and dest.stat().st_size == member.size:
                skipped += 1
            else:
                src = tf.extractfile(member)
                if src is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read())
                written += 1

            if ext == 'wav':
                filt.utts[spk] += 1
                rows.append({'filename': f'{spk}/{dest.name}', 'speaker': spk,
                             'gender': gender, 'sentence_id': idx + 1})
            if (written + skipped) % 500 == 0:
                print(f'\r  kept {written} new / {skipped} existing, '
                      f'{len(filt.taken)} speakers   ', end='', flush=True)
    except Done:
        print('\r  speaker quota reached — stopping the stream early' + ' ' * 20,
              flush=True)
    print(flush=True)
    return rows, written, skipped


def write_metadata(out_root: Path, rows: list[dict], sentences: list[str]):
    """metadata.csv joins each wav to its speaker, gender and transcription."""
    path = out_root / 'metadata.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'speaker', 'gender',
                                          'sentence_id', 'text'])
        w.writeheader()
        for row in sorted(rows, key=lambda r: r['filename']):
            sid = row['sentence_id']
            w.writerow({**row,
                        'text': sentences[sid - 1] if 0 < sid <= len(sentences) else ''})
    info(f'metadata.csv ({len(rows)} utterances) -> {path}')


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Download + filter Alcaim/CETUC (and sibling SMT-UFRJ corpora)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='corpora:\n' + '\n'.join(f'  {k:10s} {v["desc"]}'
                                        for k, v in CORPORA.items()),
    )
    ap.add_argument('--corpus', default='alcaim', choices=sorted(CORPORA),
                    help='which corpus from the mirror (default: alcaim)')
    ap.add_argument('--out', default=None,
                    help='output directory (default: {ACCENTS_BASE}/<corpus>)')
    ap.add_argument('--sentences', default='all',
                    help='all | balanced50 | balanced:<k> | 1-50 | 1,7,9 | @file (1-based)')
    ap.add_argument('--speakers', default=None,
                    help='comma-separated speaker dir names, or @file')
    ap.add_argument('--gender', default=None, choices=['m', 'f'],
                    help='keep only male or only female speakers')
    ap.add_argument('--max-speakers', type=int, default=None,
                    help='keep the first N speakers in archive order, then stop')
    ap.add_argument('--balance-gender', action='store_true',
                    help='with --max-speakers, split the quota evenly M/F')
    ap.add_argument('--max-utts', type=int, default=None,
                    help='keep at most N utterances per speaker')
    ap.add_argument('--no-transcripts', action='store_true',
                    help='skip the per-utterance .txt files')
    ap.add_argument('--keep-archive', action='store_true',
                    help='download the tarball to disk (resumable) instead of streaming')
    ap.add_argument('--archive-dir', default=None,
                    help='where to put the tarball with --keep-archive (default: --out)')
    ap.add_argument('--sentences-only', action='store_true',
                    help='fetch just the 1000-sentence transcription list and exit')
    args = ap.parse_args()

    corpus = CORPORA[args.corpus]
    out_root = resolve_out(args.out, corpus['out'])
    print(f'=== {args.corpus}: {corpus["desc"]} ===')
    print(f'  output: {out_root}')

    # Alcaim is the only corpus here with the canonical 1000-sentence list; the
    # siblings carry their own transcripts inside the archive.
    sentences: list[str] = []
    sentence_ids: set[int] | None = None
    if args.corpus == 'alcaim':
        sentences = load_sentences(out_root / 'sentences.txt')
        if args.sentences_only:
            return 0
        sentence_ids = parse_sentence_spec(args.sentences, sentences)
        if sentence_ids is not None:
            ids = sorted(i for i in sentence_ids if 0 <= i < len(sentences))
            (out_root / 'sentences_subset.txt').write_text(
                '\n'.join(sentences[i] for i in ids) + '\n', encoding='utf-8')
            (out_root / 'sentences_subset.ids').write_text(
                '\n'.join(str(i + 1) for i in ids) + '\n', encoding='utf-8')
            info(f'{len(ids)} of {len(sentences)} sentences kept '
                 f'-> sentences_subset.txt / .ids')
    elif args.sentences_only:
        warn('--sentences-only only applies to --corpus alcaim')
        return 1
    elif args.sentences != 'all':
        warn(f'--sentences is only meaningful for alcaim; ignoring for {args.corpus}')

    filt = Filter(
        speakers=parse_speakers(args.speakers),
        genders={args.gender.upper()} if args.gender else None,
        sentence_ids=sentence_ids,
        want_txt=not args.no_transcripts,
        max_utts=args.max_utts,
        max_speakers=args.max_speakers,
        balance_gender=args.balance_gender,
    )

    all_rows, totals = [], Counter()
    for url in corpus['urls']:
        print(f'  source: {url}')
        if args.keep_archive:
            archive_dir = Path(args.archive_dir) if args.archive_dir else out_root
            archive = http_download(url, archive_dir / Path(url).name)
            members = iter_archive(archive)
        else:
            members = iter_stream(url)
        rows, written, skipped = extract(members, out_root, filt)
        all_rows.extend(rows)
        totals['written'] += written
        totals['skipped'] += skipped

    if args.corpus == 'alcaim' and all_rows:
        write_metadata(out_root, all_rows, sentences)

    size = sum(f.stat().st_size for f in out_root.rglob('*') if f.is_file())
    info(f'{totals["written"]} files written, {totals["skipped"]} already present')
    info(f'{len(filt.taken)} speakers, {human(size)} in {out_root}')
    if not all_rows and not totals['skipped']:
        warn('nothing matched the filters — check --speakers / --sentences')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
