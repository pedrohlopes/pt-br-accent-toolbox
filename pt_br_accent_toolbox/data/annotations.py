from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from ..config import ANNOTATIONS_DB


DEFAULT_DB = ANNOTATIONS_DB
# Snapshot of the annotation DB that ships with the package. Used when the
# SQLite store isn't on this machine — see data/annotations/README.md.
BUNDLED_CSV = Path(__file__).resolve().parent / 'annotations' / 'annotations.csv'

MARKERS = ('s_coda', 'r_coda', 'dt_palat')


def _collect(rows, include_auto: bool) -> dict[str, dict[str, str]]:
    """
    Fold (speaker, marker, value) triples into {speaker: {marker: value}}.

    Rows must arrive newest-first: the first value seen for a speaker/marker wins,
    which is how the DB's ORDER BY updated_at DESC picks the latest annotation.
    'unsure' is dropped — it records that the annotator could not tell.
    """
    result: dict[str, dict[str, str]] = {}
    for spk, values, source in rows:
        if source == 'auto' and not include_auto:
            continue
        current = result.setdefault(spk, {})
        for marker, value in zip(MARKERS, values):
            if value and value != 'unsure' and marker not in current:
                current[marker] = value
    return result


def _from_db(db_path: Path | str, include_auto: bool):
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            'SELECT speaker, s_coda, r_coda, dt_palat, annotator FROM annotations '
            'ORDER BY updated_at DESC'
        ).fetchall()
    finally:
        conn.close()
    return _collect(
        ((r[0], r[1:4], 'auto' if r[4] == 'auto' else 'human') for r in rows),
        include_auto,
    )


def _from_csv(csv_path: Path, include_auto: bool):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # The CSV is written sorted by dataset/speaker/annotator, not by recency, so
    # sort it the same way the DB query does before folding.
    rows.sort(key=lambda r: r.get('updated_at', ''), reverse=True)
    return _collect(
        ((r['speaker'], tuple(r[m] for m in MARKERS), r.get('source', 'human'))
         for r in rows),
        include_auto,
    )


def load_annotations(db_path: Path | str | None = None,
                     include_auto: bool = False) -> dict[str, dict[str, str]]:
    """
    Load speaker annotations.

    Reads the SQLite store when it exists, otherwise falls back to the CSV
    snapshot bundled with the package, so this works on a fresh clone.

    Args:
        db_path: explicit DB path; defaults to ANNOTATIONS_DB, then the bundled CSV.
        include_auto: also return the script-written rows (annotator 'auto').
            They are excluded by default — they are not human judgements.

    Returns:
        {speaker: {marker: value}}, e.g. {'Spk1': {'s_coda': 'sibilant', ...}}.
        Only the latest annotation per speaker is kept.

    Note this is keyed by speaker ID alone, so the 115 annotated (dataset, speaker)
    pairs collapse to 111 keys: four numeric IDs occur in both brspeech_df and
    cml_tts. They are the same speakers — BRSpeech-DF's bonafide side comes from
    CML-TTS — and carry identical labels, so the merge is lossless today. Use
    `load_annotation_rows()` if you need the dataset kept apart.
    """
    path = Path(db_path or DEFAULT_DB)
    if path.exists():
        return _from_db(path, include_auto)
    if db_path is not None:
        raise FileNotFoundError(f'annotation DB not found: {path}')
    if BUNDLED_CSV.exists():
        return _from_csv(BUNDLED_CSV, include_auto)
    raise FileNotFoundError(
        f'no annotations found: neither {path} nor {BUNDLED_CSV}')


def load_annotation_rows(csv_path: Path | str | None = None) -> list[dict]:
    """
    The raw per-annotator rows from the bundled CSV, unfolded.

    Use this when you need the annotator, the dataset, the `source` column or
    disagreements between annotators — `load_annotations` throws all of that away.
    """
    path = Path(csv_path or BUNDLED_CSV)
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def get_annotated_speakers(db_path: Path | str | None = None,
                           marker: str | None = None,
                           value: str | None = None,
                           include_auto: bool = False) -> list[str]:
    """
    Get speakers with a specific annotation value.

    Args:
        marker: 's_coda' | 'r_coda' | 'dt_palat'
        value: specific annotation value, or None for any annotated

    Returns:
        list of speaker IDs
    """
    ann = load_annotations(db_path, include_auto=include_auto)
    if marker is None:
        return [spk for spk, v in ann.items() if v]

    out = []
    for spk, vals in ann.items():
        mv = vals.get(marker)
        if mv and (value is None or mv == value):
            out.append(spk)
    return out


def filter_annotations(ann: dict[str, dict[str, str]],
                       marker: str, value: str) -> dict[str, dict[str, str]]:
    """Filter annotations to speakers matching a specific marker value."""
    return {spk: v for spk, v in ann.items() if v.get(marker) == value}
