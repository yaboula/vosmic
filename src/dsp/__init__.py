"""VOSMIC - DSP Module: Preprocessing, noise gate, pitch extraction, and feature bundling."""

from src.dsp.content_encoder import ContentEncoder
from src.dsp.dsp_thread import DSPThread
from src.dsp.feature_bundle import FeatureBundle, FeatureBundleAssembler
from src.dsp.noise_gate import NoiseGate
from src.dsp.normalizer import LoudnessNormalizer
from src.dsp.pitch_extractor import PitchExtractor

__all__ = [
    "ContentEncoder",
    "DSPThread",
    "FeatureBundle",
    "FeatureBundleAssembler",
    "LoudnessNormalizer",
    "NoiseGate",
    "PitchExtractor",
]
