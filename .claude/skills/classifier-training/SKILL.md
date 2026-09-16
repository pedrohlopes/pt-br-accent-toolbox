---
name: classifier-training
description: Training and evaluation of classifiers on pt_br_accent_toolbox feature vectors — Leave-One-Speaker-Out cross-validation, feature x classifier ablation grids, model persistence, and speaker annotation loading. Used when asked to train a classifier, run LOSO-CV, run an ablation grid, save/load a trained model, or classify accent markers in this repo.
---

# Classifier Training & Evaluation

**Working directory**: this checkout lives at `tools/pt-br-accent-toolbox/`
(hyphenated) so it can never shadow the `pt_br_accent_toolbox` package name —
`from pt_br_accent_toolbox import ...` now works from any cwd, including
`/mnt/data/accents`.

`pt_br_accent_toolbox.classification` trains/evaluates classifiers on the feature
vectors produced by the `acoustic-features` / `ssl-embeddings` skills, using
Leave-One-Speaker-Out (LOSO) cross-validation — the standard protocol for this
project since speakers, not clips, are the unit of generalization.

## LOSO cross-validation

```python
import numpy as np
from pt_br_accent_toolbox.classification.loso import loso_cv, compute_eer
from sklearn.ensemble import RandomForestClassifier

X = np.load("results/formants.npz")["vector"]   # (N, D)
y = np.array([...])                              # (N,) binary/multi-class labels
groups = np.array([...])                         # (N,) speaker IDs — one fold per unique ID

res = loso_cv(X, y, groups, RandomForestClassifier(n_estimators=300))
print(res['acc'])                # mean accuracy across folds
print(res['per_speaker'])        # {speaker_id: {'acc', 'n', 'preds', 'true', 'scores'}}
print(compute_eer(res['labels'], res['scores']))   # EER %, binary tasks only
```

`impute_strategy` (default `'nanmean'`) controls how `NaN` feature values (e.g.
missing vowels in `formants`) are handled: `'nanmean'` fills with the training
fold's finite-row mean, `'drop'` removes rows with any `NaN`, `'none'` fills with 0.
Fit the imputation **inside each LOSO fold** (which `loso_cv` already does) — never
impute on the full dataset before splitting, that leaks test-speaker statistics
into training.

## Ablation grid (feature × classifier)

```python
from pt_br_accent_toolbox.classification.ablation import ablation_grid, CLASSIFIERS

# CLASSIFIERS: 'rf', 'gb', 'lr', 'svm', 'xgb' — sklearn-compatible factories,
# call CLASSIFIERS['xgb']() to get a fresh instance with tuned defaults

results = ablation_grid(
    feature_sets={"formants": X_formants, "mfcc": X_mfcc, "ssl_ecapa": X_ecapa},
    labels=y,
    groups=groups,
    classifier_names=["rf", "xgb"],   # omit for all 5
)
for r in results:
    print(f"{r['feature']:12s} {r['classifier']:4s} → acc={r['acc']:.3f}")
```

`xgb` uses the parameters that produced this project's actual published results:
`max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.9,
colsample_bytree=0.9, reg_lambda=1.0, eval_metric='logloss', tree_method='hist'`.

## Saving and loading a trained model

```python
from pt_br_accent_toolbox.classification.persistence import save_model, load_model

clf = CLASSIFIERS["xgb"]().fit(X, y)
save_model("models/s_coda", clf, meta={
    "classes": sorted(set(y.tolist())),
    "feat_names": [...],
    "loso_acc": res["acc"],
})
# writes models/s_coda/model.pkl + models/s_coda/meta.json

clf2, meta = load_model("models/s_coda")
probs = clf2.predict_proba(X_new)
```

Never assume probability-column order matches label order from memory — read
`meta['classes']` back out and index against it.

## Labels: loading speaker annotations

```python
from pt_br_accent_toolbox.data.annotations import load_annotations, get_annotated_speakers

all_ann = load_annotations()      # {'Spk1': {'s_coda': 'chiado', ...}, ...}
                                    # reads $ANNOTATIONS_DB (default {ACCENTS_BASE}/classifier_ui/annotations.db)

s_speakers = get_annotated_speakers(marker="s_coda", value="chiado")
```

CLI: `pt-br-accent-toolbox annotations --marker s_coda --value chiado`

Only the most recent annotation per speaker is kept (`ORDER BY updated_at DESC`);
rows with value `'unsure'` are dropped.

## Evaluation protocol / pitfalls

- **LOSO, not k-fold**: always group by speaker ID, never by clip — a classifier
  that sees a speaker's other clips in training will trivially overfit to speaker
  identity rather than the phonological marker.
- **No leakage on thresholds**: any decision threshold (percentile cut, univariate
  cutoff) must be chosen on training folds only, never on the full pool — this
  project has previously seen in-pool metrics collapse when moved to a genuinely
  held-out set (a `~95% → ~20%` swing was traced to exactly this).
- **Report both clip- and speaker-level accuracy** where the unit of prediction is
  a clip: clip-level accuracy from `res['acc']`, speaker-level majority-vote from
  aggregating `res['per_speaker']`.
