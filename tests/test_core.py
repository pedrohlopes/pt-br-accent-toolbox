"""Import and smoke tests for pt_br_accent_toolbox.

Run with:  python -m pytest tests/ -v
"""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent


def test_imports():
    """All public API imports resolve."""
    from pt_br_accent_toolbox import (
        FeaturePipeline,
        loso_cv,
        compute_eer,
        ablation_grid,
        CLASSIFIERS,
        load_annotations,
        get_annotated_speakers,
    )
    assert FeaturePipeline is not None
    assert loso_cv is not None
    assert compute_eer is not None
    assert ablation_grid is not None
    assert isinstance(CLASSIFIERS, dict)
    assert len(CLASSIFIERS) == 4


def test_config():
    """Config constants are accessible."""
    from pt_br_accent_toolbox.config import (
        SR, FRAME_MS, PHONE_GROUPS, VOWEL_SET,
        PX_MODEL, SSL_MODELS,
    )
    assert SR == 16000
    assert set(PHONE_GROUPS.keys()) == {'s', 'r', 'dt'}
    assert len(VOWEL_SET) == 7
    assert len(SSL_MODELS) == 4


def test_phone_groups():
    """Phone group contents are reasonable."""
    from pt_br_accent_toolbox.config import PHONE_GROUPS
    assert len(PHONE_GROUPS['s']) >= 3
    assert len(PHONE_GROUPS['r']) >= 4
    assert len(PHONE_GROUPS['dt']) >= 3


def test_classifier_factories():
    """Classifier factories produce working sklearn estimators."""
    from pt_br_accent_toolbox.classification.ablation import CLASSIFIERS
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier

    for name, factory in CLASSIFIERS.items():
        clf = factory()
        assert hasattr(clf, 'fit'), f'{name} has no fit method'


def test_spectral_moments():
    """Spectral moments return 6D float32 vector for any audio."""
    import numpy as np
    from pt_br_accent_toolbox.features.spectral import spectral_moments

    # Synthetic 16 kHz audio: 1 second of white noise
    audio = np.random.randn(16000).astype(np.float32)
    feat = spectral_moments(audio)
    assert feat.shape == (6,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))

    # Silent audio should return zeros
    silence = np.zeros(16000, np.float32)
    feat_sil = spectral_moments(silence)
    assert feat_sil.shape == (6,)


def test_spectral_edge_cases():
    """Spectral moments handle short/empty clips."""
    import numpy as np
    from pt_br_accent_toolbox.features.spectral import spectral_moments

    # Very short clip (smaller than N_FFT)
    short = np.random.randn(256).astype(np.float32)
    feat = spectral_moments(short)
    assert feat.shape == (6,)
    assert np.all(np.isfinite(feat))

    # Zero-length should not crash
    empty = np.array([], np.float32)
    feat = spectral_moments(empty)
    assert feat.shape == (6,)


def test_mfcc():
    """MFCC extraction returns expected shape."""
    import numpy as np
    from pt_br_accent_toolbox.features.mfcc import extract_mfcc, speaker_mfcc_mean

    audio = np.random.randn(32000).astype(np.float32)
    mfcc = extract_mfcc(audio, sr=16000, n_mfcc=13)
    assert mfcc.shape[0] == 13  # n_mfcc
    assert mfcc.dtype == np.float32

    # Speaker mean with no valid paths returns zeros
    with_mock = speaker_mfcc_mean([None])
    assert with_mock.shape == (13,)
    assert np.allclose(with_mock, 0.0)  # zeros when no audio


def test_loso_cv_structure():
    """LOSO CV returns the expected keys."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from pt_br_accent_toolbox.classification.loso import loso_cv

    X = np.random.randn(20, 6).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)
    groups = np.array([f'spk_{i}' for i in range(20)])

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    res = loso_cv(X, y, groups, clf, impute_strategy='none')

    expected_keys = {'scores', 'labels', 'groups', 'preds', 'acc', 'per_speaker'}
    assert expected_keys.issubset(res.keys())
    assert isinstance(res['acc'], float)


def test_eer():
    """EER computation returns 0-100 range."""
    import numpy as np
    from pt_br_accent_toolbox.classification.loso import compute_eer

    # Perfect separation
    y = np.array([0, 0, 1, 1])
    s = np.array([-2.0, -1.0, 1.0, 2.0])
    eer = compute_eer(y, s)
    assert 0.0 <= eer <= 100.0

    # All same score (worst case)
    s2 = np.array([0.0, 0.0, 0.0, 0.0])
    eer2 = compute_eer(y, s2)
    assert 0.0 <= eer2 <= 100.0


def test_ablation_grid():
    """Ablation grid returns list of results."""
    import numpy as np
    from pt_br_accent_toolbox.classification.ablation import ablation_grid

    X = {'test_feat': np.random.randn(12, 6).astype(np.float32)}
    y = np.array([0] * 6 + [1] * 6)
    groups = np.array([f'spk_{i}' for i in range(12)])

    results = ablation_grid(X, y, groups, classifier_names=['rf'])
    assert len(results) == 1
    assert 'acc' in results[0]
    assert results[0]['feature'] == 'test_feat'
    assert results[0]['classifier'] == 'rf'
