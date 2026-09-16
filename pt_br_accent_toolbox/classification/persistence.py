from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_model(out_dir: Path | str, clf: Any, meta: dict) -> None:
    """
    Save a fitted classifier and its metadata to a directory.

    Args:
        out_dir: directory to write model.pkl and meta.json into (created if missing)
        clf: fitted sklearn-compatible classifier (must support pickle)
        meta: JSON-serializable metadata — e.g. {'classes': [...], 'feat_names': [...],
              'loso_acc': 0.79, 'class_counts': {...}}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def load_model(model_dir: Path | str) -> tuple[Any, dict]:
    """
    Load a classifier and metadata previously saved with save_model().

    Returns:
        (clf, meta)
    """
    model_dir = Path(model_dir)
    with open(model_dir / 'model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(model_dir / 'meta.json') as f:
        meta = json.load(f)
    return clf, meta
