"""
Export the speaker-annotation store to the CSVs bundled with the package.

The annotations live in a SQLite database written by the Flask annotation UI
(`classifier_ui/annotations.db`, outside this repo). That database is the source
of truth; this script snapshots it into
`pt_br_accent_toolbox/data/annotations/`, which ships with the package so a fresh
clone can reproduce the labelled cohort without access to the original DB.

Outputs:
    annotations.csv       one row per (annotator, dataset, speaker)
    speakers.csv          one row per (dataset, speaker) — the majority label per
                          marker across annotators, plus agreement counts
    summary.csv           label counts per dataset x marker x value
    todo.csv              speakers with at least one marker still unlabelled —
                          the queue to work through in the annotation UI

Every row carries `source` = human | auto, so a consumer can always tell a person's
judgement from a script's. There are currently no `auto` rows; the column stays
because the annotation UI can still write them.

Usage:
    python tools/export_annotations.py
    python tools/export_annotations.py --db /path/to/annotations.db
    python tools/export_annotations.py --out /tmp/ann --stdout
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import accents_base, info, warn  # noqa: E402

MARKERS = ('s_coda', 'r_coda', 'dt_palat')
AUTO_ANNOTATORS = {'auto'}
BUNDLED = (Path(__file__).resolve().parent.parent
           / 'pt_br_accent_toolbox' / 'data' / 'annotations')


def default_db() -> Path:
    return accents_base() / 'classifier_ui' / 'annotations.db'


def read_rows(db: Path) -> list[dict]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        'SELECT annotator, dataset, speaker, s_coda, r_coda, dt_palat, '
        '       notes, gold, updated_at '
        'FROM annotations ORDER BY dataset, speaker, annotator')]
    conn.close()

    for r in rows:
        r['source'] = 'auto' if r['annotator'] in AUTO_ANNOTATORS else 'human'
        for m in MARKERS:
            # 'unsure' is the UI's explicit "I could not tell" — keep it distinct
            # from a marker that was simply never labelled.
            r[m] = (r[m] or '').strip()
        r['notes'] = (r['notes'] or '').strip()
    return rows


def write_annotations(rows: list[dict], out: Path) -> Path:
    path = out / 'annotations.csv'
    fields = ['dataset', 'speaker', 'annotator', 'source',
              's_coda', 'r_coda', 'dt_palat', 'gold', 'notes', 'updated_at']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return path


def consensus(rows: list[dict], out: Path) -> Path:
    """
    One row per (dataset, speaker): the majority human label per marker.

    `n_annotators` counts humans who labelled that speaker at all; `agree_<marker>`
    is how many of them chose the winning value. A tie leaves the label empty and
    sets agree to 0 — there are only a handful of multiply-annotated speakers, so
    ties are resolved by hand rather than by an arbitrary rule.
    """
    by_speaker: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[(r['dataset'], r['speaker'])].append(r)

    path = out / 'speakers.csv'
    fields = (['dataset', 'speaker', 'source', 'n_annotators', 'annotators']
              + [c for m in MARKERS for c in (m, f'agree_{m}')])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (dataset, speaker), items in sorted(by_speaker.items()):
            humans = [r for r in items if r['source'] == 'human']
            pool = humans or items
            row = {
                'dataset': dataset,
                'speaker': speaker,
                'source': 'human' if humans else 'auto',
                'n_annotators': len(humans),
                'annotators': '; '.join(sorted({r['annotator'] for r in pool})),
            }
            for m in MARKERS:
                votes = Counter(r[m] for r in pool if r[m])
                if not votes:
                    row[m], row[f'agree_{m}'] = '', 0
                    continue
                ranked = votes.most_common()
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    row[m], row[f'agree_{m}'] = '', 0   # tie — resolve by ear
                else:
                    row[m], row[f'agree_{m}'] = ranked[0][0], ranked[0][1]
            w.writerow(row)
    return path


def summarize(rows: list[dict], out: Path) -> Path:
    path = out / 'summary.csv'
    counts: Counter = Counter()
    for r in rows:
        for m in MARKERS:
            if r[m]:
                counts[(r['dataset'], m, r[m], r['source'])] += 1
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'marker', 'value', 'source', 'n'])
        for (dataset, marker, value, source), n in sorted(counts.items()):
            w.writerow([dataset, marker, value, source, n])
    return path


def write_todo(rows: list[dict], out: Path) -> Path:
    """Speakers with at least one marker still unlabelled, for the annotation UI."""
    by_speaker: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[(r['dataset'], r['speaker'])].append(r)

    path = out / 'todo.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'speaker', 'missing', 'have', 'annotators'])
        n = 0
        for (dataset, speaker), items in sorted(by_speaker.items()):
            labelled = {m for m in MARKERS if any(r[m] for r in items)}
            missing = [m for m in MARKERS if m not in labelled]
            if not missing:
                continue
            w.writerow([dataset, speaker, ' '.join(missing),
                        ' '.join(sorted(labelled)),
                        '; '.join(sorted({r['annotator'] for r in items}))])
            n += 1
    info(f'{n} speakers still need at least one marker')
    return path


def print_summary(rows: list[dict]):
    humans = [r for r in rows if r['source'] == 'human']
    auto = [r for r in rows if r['source'] == 'auto']
    speakers = {(r['dataset'], r['speaker']) for r in rows}
    print(f'  {len(rows)} rows ({len(humans)} human, {len(auto)} auto), '
          f'{len(speakers)} speakers')

    per_ds = Counter(r['dataset'] for r in rows)
    for ds, n in per_ds.most_common():
        spk = len({r['speaker'] for r in rows if r['dataset'] == ds})
        print(f'    {ds:18s} {n:3d} rows  {spk:3d} speakers')

    for m in MARKERS:
        votes = Counter(r[m] for r in humans if r[m])
        print(f'    {m:10s} ' + '  '.join(f'{v}={n}' for v, n in votes.most_common()))

    multi = [k for k, v in Counter(
        (r['dataset'], r['speaker']) for r in humans).items() if v > 1]
    print(f'    {len(multi)} speakers labelled by more than one annotator')


def main():
    ap = argparse.ArgumentParser(description='Export annotations to CSV')
    ap.add_argument('--db', default=None,
                    help='annotations.db (default: {ACCENTS_BASE}/classifier_ui/annotations.db)')
    ap.add_argument('--out', default=None,
                    help=f'output directory (default: {BUNDLED})')
    ap.add_argument('--stdout', action='store_true',
                    help='also print the summary table')
    args = ap.parse_args()

    db = Path(args.db) if args.db else default_db()
    if not db.exists():
        warn(f'annotation DB not found at {db}')
        warn('pass --db, or set ACCENTS_BASE to the directory holding classifier_ui/')
        return 1

    out = Path(args.out) if args.out else BUNDLED
    out.mkdir(parents=True, exist_ok=True)

    print(f'=== annotations: {db} ===')
    rows = read_rows(db)
    print_summary(rows)

    for path in (write_annotations(rows, out), consensus(rows, out),
                 summarize(rows, out), write_todo(rows, out)):
        info(f'{path.name} -> {path}')
    if args.stdout:
        print((out / 'summary.csv').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
