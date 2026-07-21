from __future__ import annotations

import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .loso import loso_cv

CLASSIFIERS = {
    'rf': lambda **kw: RandomForestClassifier(n_estimators=300, random_state=42, **kw),
    'gb': lambda **kw: GradientBoostingClassifier(n_estimators=300, random_state=42, **kw),
    'lr': lambda **kw: LogisticRegression(max_iter=5000, random_state=42, **kw),
    'svm': lambda **kw: SVC(probability=True, random_state=42, **kw),
}


def ablation_grid(
    feature_sets: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    classifier_names: list[str] | None = None,
    fit_params: dict | None = None,
    impute_strategy: str = 'nanmean',
) -> list[dict]:
    """
    Run full feature x classifier ablation grid with LOSO CV.

    Args:
        feature_sets: {name: (N, D)} feature matrices (same N, same row order)
        labels: (N,) target labels
        groups: (N,) speaker IDs
        classifier_names: subset of CLASSIFIERS keys, or all
        fit_params: extra params for clf.fit()
        impute_strategy: passed to loso_cv

    Returns:
        list of result dicts with 'feature', 'classifier', 'acc', etc.
    """
    classifier_names = classifier_names or list(CLASSIFIERS.keys())
    results = []

    for fname, clf_name in product(feature_sets.keys(), classifier_names):
        X = feature_sets[fname]
        clf_factory = CLASSIFIERS[clf_name]
        clf = clf_factory()

        res = loso_cv(X, labels, groups, clf, fit_params, impute_strategy)
        results.append({
            'feature': fname,
            'classifier': clf_name,
            'acc': res['acc'],
            'n_train': len(np.unique(groups)),
            **{k: v for k, v in res.items() if k not in ('per_speaker',)},
        })
    return results
