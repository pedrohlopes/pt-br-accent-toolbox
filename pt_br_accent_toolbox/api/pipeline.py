from __future__ import annotations

"""
High-level feature extraction pipeline.

Provides a unified interface to extract any combination of features
(phoneticxeus, formants, spectral, ssl, mfcc, zipa) for a set of speakers,
given their audio file lists.
"""

import numpy as np
from typing import Any

from ..config import SR, PHONE_GROUPS
from ..alignment.zipa import (
    load_audio, load_vocab, make_fbank_extractor, make_session,
    utterance_logprobs, get_spikes, extract_marker_frames,
)
from ..features.spectral import spectral_moments
from ..features.phoneticxeus import extract_for_markers as px_extract, speaker_vector as px_speaker
from ..features.formants import extract_for_speaker as formant_extract
from ..features.mfcc import speaker_mfcc_mean
from ..features.ssl import speaker_ssl_vector


class FeaturePipeline:
    """
    Extract features for a set of speakers.

    Usage:
        pipeline = FeaturePipeline()
        results = pipeline.extract(
            speakers={'spk1': [path1, path2], 'spk2': [path3]},
            features=['px', 'formants', 'mfcc'],
        )
        # results: {feature_name: {speaker: np.array}}
    """

    def __init__(self, zipa_model=None, zipa_tokens=None):
        self._sess = None
        self._extractor = None
        self._vocab = None
        self._px_model = None
        self._px_device = None
        self._px_token_list = None
        self._zipa_model = zipa_model
        self._zipa_tokens = zipa_tokens

    def _ensure_zipa(self):
        if self._sess is None:
            self._sess = make_session(self._zipa_model)
            self._extractor = make_fbank_extractor()
            self._vocab = load_vocab(self._zipa_tokens)

    def _ensure_px(self):
        if self._px_model is None:
            from ..alignment.phoneticxeus import load_model as load_px
            self._px_model, self._px_device, self._px_token_list = load_px()

    def get_markers(self, audio: np.ndarray) -> list[dict]:
        """Get phonological marker frames from audio."""
        self._ensure_zipa()
        return extract_marker_frames(audio, self._sess, self._extractor)

    def extract_speaker(
        self,
        audio_paths: list[str],
        features: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """
        Extract requested features for one speaker.

        Args:
            audio_paths: list of audio file paths
            features: subset of ['spec', 'zipa', 'px', 'formants', 'mfcc', 'ssl_ecapa', 'ssl_xlsr', 'ssl_hubert', 'ssl_w2vbert']
            groups: override PHONE_GROUPS for px/zipa

        Returns:
            {feature_name: (vector, metadata)}
        """
        features = features or ['spec', 'zipa', 'px', 'formants', 'mfcc']
        result: dict[str, Any] = {}

        # Collect markers from all utterances
        all_markers: list[dict] = []
        valid_audios: list[np.ndarray] = []
        self._ensure_zipa()

        for p in audio_paths:
            try:
                audio = load_audio(p)[:12 * SR]
            except Exception:
                continue
            audio = np.nan_to_num(audio).astype(np.float32)
            if len(audio) < 512:
                continue
            markers = extract_marker_frames(audio, self._sess, self._extractor)
            all_markers.extend(markers)
            valid_audios.append(audio)

        # Spectral moments (per-marker)
        if 'spec' in features:
            clips = []
            for m in all_markers:
                center = int(m['time_s'] * SR)
                for audio in valid_audios:
                    lo = max(0, center - 320)
                    hi = min(len(audio), center + 320)
                    clips.append(spectral_moments(audio[lo:hi]))
            if clips:
                result['spec'] = (np.mean(clips, axis=0), {'n_clips': len(clips)})

        # ZIPA logits at markers
        if 'zipa' in features and all_markers:
            from ..features.zipa import extract_for_markers as zipa_fm
            lp = utterance_logprobs(valid_audios[0], self._sess, self._extractor)
            tid = {v: k for k, v in self._vocab.items()}
            feat = zipa_fm(valid_audios[0], all_markers, lp, tid, groups=groups)
            if feat.size > 0:
                result['zipa'] = (feat, {'n_markers': len(all_markers)})

        # PhoneticXeus
        if 'px' in features and all_markers:
            self._ensure_px()
            for audio in valid_audios:
                feat = px_extract(audio, all_markers, self._px_model, self._px_device,
                                  self._px_token_list, groups=groups)
                if feat is not None and feat.size > 0:
                    result['px'] = (feat, {'n_markers': len(all_markers)})
                    break

        # Formants
        if 'formants' in features:
            vec, nvow = formant_extract(audio_paths, self._sess, self._extractor,
                                        self._zipa_model, self._zipa_tokens)
            result['formants'] = (vec, {'n_vowels': nvow})

        # MFCC
        if 'mfcc' in features:
            vec = speaker_mfcc_mean(audio_paths)
            result['mfcc'] = (vec, {})

        # SSL
        for ssl_name in ['ecapa', 'xlsr', 'hubert', 'w2vbert']:
            if f'ssl_{ssl_name}' in features:
                vec, nutt = speaker_ssl_vector(audio_paths, ssl_name)
                result[f'ssl_{ssl_name}'] = (vec, {'n_utterances': nutt})

        return result

    def extract(
        self,
        speakers: dict[str, list[str]],
        features: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Extract features for multiple speakers.

        Args:
            speakers: {speaker_id: [audio_paths]}
            features: feature names to extract
            groups: override PHONE_GROUPS

        Returns:
            {feature_name: {speaker_id: vector}}
        """
        results: dict[str, dict[str, np.ndarray]] = {}

        for spk, paths in speakers.items():
            res = self.extract_speaker(paths, features, groups)
            for fname, (vec, meta) in res.items():
                results.setdefault(fname, {})
                results[fname][spk] = vec

        return results
