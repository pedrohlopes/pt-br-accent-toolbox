"""
End-to-end example: extract features from audio and run LOSO classification.

This demonstrates the full pipeline:
  1. Load audio files for a set of speakers
  2. Extract a mix of marker-local, vowel, and global features
  3. Train and evaluate a classifier with LOSO cross-validation
  4. Run an ablation grid to compare feature sets

Usage:
    python examples/run_pipeline.py --audio-dir /path/to/speaker_dirs

Audio directory structure:
    /path/to/speaker_dirs/
        spk_001/
            utt1.wav
            utt2.wav
        spk_002/
            utt1.wav
            ...
"""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def find_audio_speakers(audio_dir: Path, min_wavs: int = 2) -> dict[str, list[str]]:
    """Scan a directory for speaker subdirectories with WAV files."""
    speakers = {}
    for spk_dir in sorted(audio_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        wavs = sorted(spk_dir.glob("*.wav"))
        if len(wavs) >= min_wavs:
            speakers[spk_dir.name] = [str(w) for w in wavs]
    return speakers


def main():
    parser = argparse.ArgumentParser(description="PT-BR Accent Toolbox demo")
    parser.add_argument("--audio-dir", required=True,
                        help="Directory with speaker subfolders containing WAVs")
    parser.add_argument("--min-wavs", type=int, default=2,
                        help="Minimum WAVs per speaker (default: 2)")
    parser.add_argument("--features", nargs="+",
                        default=["spec", "mfcc", "formants"],
                        help="Features to extract")
    parser.add_argument("--classifier", default="rf",
                        choices=["rf", "gb", "lr", "svm"],
                        help="Classifier for LOSO (default: rf)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        sys.exit(1)

    # ── 1. Discover speakers ────────────────────────────────────────────
    speakers = find_audio_speakers(audio_dir, args.min_wavs)
    if not speakers:
        print(f"No speakers found in {audio_dir} "
              f"(need subdirectories with ≥{args.min_wavs} WAVs each)")
        sys.exit(1)
    print(f"Found {len(speakers)} speakers:")
    for spk, wavs in list(speakers.items())[:5]:
        print(f"  {spk}: {len(wavs)} files")
    if len(speakers) > 5:
        print(f"  ... and {len(speakers) - 5} more")

    # ── 2. Extract features ─────────────────────────────────────────────
    from pt_br_accent_toolbox.api.pipeline import FeaturePipeline

    print(f"\nExtracting features: {args.features}")
    pipe = FeaturePipeline()
    results = pipe.extract(speakers, features=args.features)

    if not results:
        print("No features extracted. Check that ZIPA model is available.")
        sys.exit(1)

    for fname, spk_dict in results.items():
        print(f"  {fname}: {len(spk_dict)} speakers, "
              f"dim={list(spk_dict.values())[0].shape[0]}")

    # ── 3. LOSO classification example ──────────────────────────────────
    # Requires labels. Demo creates synthetic labels if none provided.
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from pt_br_accent_toolbox.classification.loso import loso_cv, compute_eer
    from pt_br_accent_toolbox.classification.ablation import ablation_grid, CLASSIFIERS

    spk_ids = sorted(speakers.keys())
    labels = np.array([0 if i % 2 == 0 else 1 for i in range(len(spk_ids))])

    # Build feature matrix from first extracted feature
    first_feat = list(results.keys())[0]
    spk_order = list(results[first_feat].keys())
    X = np.stack([results[first_feat][s] for s in spk_order])

    # Impute NaN (speakers with missing features)
    X_imp = np.nan_to_num(X, nan=np.nanmean(X[np.isfinite(X).all(axis=1)], axis=0))

    print(f"\nLOSO CV on {first_feat} ({X_imp.shape[1]}D, "
          f"{len(spk_ids)} speakers):")
    clf = CLASSIFIERS[args.classifier]()
    res = loso_cv(X_imp, labels, np.array(spk_ids), clf)
    print(f"  Accuracy: {res['acc']:.3f}")
    print(f"  EER: {compute_eer(res['labels'], res['scores']):.1f}%")

    # ── 4. Ablation grid (if multiple features) ─────────────────────────
    if len(results) >= 2:
        print(f"\nAblation grid across {len(results)} feature sets:")
        feature_sets = {}
        for fname, spk_dict in results.items():
            spk_ordered = list(spk_dict.keys())
            feature_sets[fname] = np.stack([spk_dict[s] for s in spk_ordered])
        abl_results = ablation_grid(
            feature_sets=feature_sets,
            labels=labels,
            groups=np.array(spk_ids),
            classifier_names=[args.classifier],
        )
        for r in abl_results:
            print(f"  {r['feature']:12s}  → acc={r['acc']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
