from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score


def loso_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    clf, fit_params: dict | None = None,
    impute_strategy: str = 'nanmean',
) -> dict:
    """
    Leave-One-Speaker-Out cross-validation.

    Args:
        X: (N, D) feature matrix
        y: (N,) labels
        groups: (N,) speaker IDs
        clf: sklearn-compatible classifier
        fit_params: extra params for clf.fit()
        impute_strategy: 'nanmean' | 'drop' | 'none'

    Returns:
        dict with 'scores', 'labels', 'groups', 'acc', 'per_speaker'
    """
    logo = LeaveOneGroupOut()
    scores, preds, true, group_ids = [], [], [], []
    per_speaker = {}

    if impute_strategy == 'nanmean':
        finite_mask = np.isfinite(X).all(axis=1)
        nat_mean = np.nanmean(X[finite_mask], axis=0) if finite_mask.any() else np.nanmean(X, axis=0)
        nat_mean = np.where(np.isfinite(nat_mean), nat_mean, 0.0)

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_te = groups[test_idx]
        spk = g_te[0]

        if impute_strategy == 'nanmean':
            X_tr_imp = np.nan_to_num(X_tr, nan=nat_mean)
            X_te_imp = np.nan_to_num(X_te, nan=nat_mean)
        elif impute_strategy == 'drop':
            mask_tr = np.isfinite(X_tr).all(axis=1)
            mask_te = np.isfinite(X_te).all(axis=1)
            X_tr_imp, y_tr = X_tr[mask_tr], y_tr[mask_tr]
            X_te_imp, y_te, g_te = X_te[mask_te], y_te[mask_te], g_te[mask_te]
        else:
            X_tr_imp = np.nan_to_num(X_tr, nan=0.0)
            X_te_imp = np.nan_to_num(X_te, nan=0.0)

        if len(X_tr_imp) == 0 or len(X_te_imp) == 0:
            continue

        clf_instance = type(clf)(**getattr(clf, 'get_params', lambda: {})())
        fp = fit_params or {}
        clf_instance.fit(X_tr_imp, y_tr, **fp)

        te_scores = clf_instance.decision_function(X_te_imp) if hasattr(clf_instance, 'decision_function') else clf_instance.predict_proba(X_te_imp)[:, 1] if hasattr(clf_instance, 'predict_proba') and len(np.unique(y)) == 2 else np.zeros(len(X_te_imp))
        te_preds = clf_instance.predict(X_te_imp)

        scores.extend(te_scores.tolist())
        preds.extend(te_preds.tolist())
        true.extend(y_te.tolist())
        group_ids.extend(g_te.tolist())

        acc = accuracy_score(y_te, te_preds)
        per_speaker[spk] = {'acc': acc, 'n': len(y_te), 'preds': te_preds.tolist(),
                            'true': y_te.tolist(), 'scores': te_scores.tolist()}

    return {
        'scores': np.array(scores),
        'labels': np.array(true),
        'groups': np.array(group_ids),
        'preds': np.array(preds),
        'acc': accuracy_score(true, preds) if true else np.nan,
        'per_speaker': per_speaker,
    }


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute EER from binary labels and detection scores."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    thresholds = np.sort(scores)
    eer = 1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        fa = np.mean(pred[y_true == 0])
        fm = 1.0 - np.mean(pred[y_true == 1])
        eer = min(eer, abs(fa + fm) / 2) if abs(fa - fm) < 0.01 else eer
    return eer * 100
