from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import ANNOTATIONS_DB


DEFAULT_DB = ANNOTATIONS_DB


def load_annotations(db_path: Path | str | None = None) -> dict[str, dict[str, str]]:
    """
    Load speaker annotations from SQLite DB.

    Returns:
        {speaker: {marker: value}} e.g. {'Spk1': {'s_coda': 'sibilant', 'r_coda': 'caipira', ...}}
        Only keeps latest annotation per speaker (ORDER BY updated_at DESC).
    """
    conn = sqlite3.connect(str(db_path or DEFAULT_DB))
    result: dict[str, dict[str, str]] = {}

    rows = conn.execute(
        'SELECT speaker, s_coda, r_coda, dt_palat FROM annotations '
        'ORDER BY updated_at DESC'
    ).fetchall()

    for spk, s_coda, r_coda, dt_palat in rows:
        if spk not in result:
            result[spk] = {}
        if s_coda and s_coda != 'unsure':
            result[spk]['s_coda'] = s_coda
        if r_coda and r_coda != 'unsure':
            result[spk]['r_coda'] = r_coda
        if dt_palat and dt_palat != 'unsure':
            result[spk]['dt_palat'] = dt_palat

    conn.close()
    return result


def get_annotated_speakers(db_path: Path | str | None = None,
                            marker: str | None = None,
                            value: str | None = None) -> list[str]:
    """
    Get speakers with a specific annotation value.

    Args:
        marker: 's_coda' | 'r_coda' | 'dt_palat'
        value: specific annotation value, or None for any annotated

    Returns:
        list of speaker IDs
    """
    ann = load_annotations(db_path)
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
