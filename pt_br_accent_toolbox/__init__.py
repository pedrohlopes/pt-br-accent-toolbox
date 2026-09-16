"""
pt_br_accent_toolbox — feature extraction and analysis tools for Brazilian Portuguese speech.

Generic framework for phoneme alignment, phonological marker detection,
and multi-source feature extraction (spectral, ZIPA, PhoneticXeus, formants, MFCC, SSL).
"""

__version__ = '0.1.0'

from .api.pipeline import FeaturePipeline
from .classification.loso import loso_cv, compute_eer
from .classification.ablation import ablation_grid, CLASSIFIERS
from .classification.persistence import save_model, load_model
from .data.annotations import load_annotations, get_annotated_speakers
