"""
test_features.py — Unit Tests for Feature Extraction
======================================================
Tests the feature extraction pipeline in src/features.py.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import (
    load_audio,
    extract_features,
    extract_features_from_file,
    SAMPLE_RATE,
    DURATION,
)


def _create_test_wav(filepath: str, duration: float = 1.0,
                     sr: int = 22050) -> None:
    """Generate a short synthetic WAV file for testing."""
    import soundfile as sf
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(filepath, audio, sr)


class TestLoadAudio:
    """Tests for the load_audio function."""

    def test_returns_correct_length(self, tmp_path):
        """Audio should be padded/truncated to exactly DURATION * SR."""
        wav_path = str(tmp_path / "test.wav")
        _create_test_wav(wav_path, duration=1.0)

        y = load_audio(wav_path)
        expected_length = int(SAMPLE_RATE * DURATION)
        assert len(y) == expected_length

    def test_short_audio_is_padded(self, tmp_path):
        """Files shorter than DURATION should be zero-padded."""
        wav_path = str(tmp_path / "short.wav")
        _create_test_wav(wav_path, duration=0.5)

        y = load_audio(wav_path)
        expected_length = int(SAMPLE_RATE * DURATION)
        assert len(y) == expected_length

    def test_returns_numpy_array(self, tmp_path):
        """Output must be a NumPy array."""
        wav_path = str(tmp_path / "test.wav")
        _create_test_wav(wav_path, duration=1.0)

        y = load_audio(wav_path)
        assert isinstance(y, np.ndarray)


class TestExtractFeatures:
    """Tests for the extract_features function."""

    def test_feature_vector_shape(self):
        """Feature vector should have 60 dimensions."""
        y = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)
        features = extract_features(y)
        assert features.shape == (60,)

    def test_features_are_finite(self):
        """All feature values should be finite (no NaN/inf)."""
        y = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)
        features = extract_features(y)
        assert np.all(np.isfinite(features))


class TestExtractFeaturesFromFile:
    """Tests for the extract_features_from_file wrapper."""

    def test_end_to_end(self, tmp_path):
        """Full pipeline: file → feature vector."""
        wav_path = str(tmp_path / "test.wav")
        _create_test_wav(wav_path, duration=2.0)

        features = extract_features_from_file(wav_path)
        assert features.shape == (60,)
        assert np.all(np.isfinite(features))

    def test_invalid_file_raises(self, tmp_path):
        """An invalid file should raise an exception."""
        bad_path = str(tmp_path / "nonexistent.wav")
        with pytest.raises(Exception):
            extract_features_from_file(bad_path)


class TestPreprocessLabelParsing:
    """Tests for RAVDESS filename label parsing."""

    def test_emotion_mapping(self):
        """Check that known RAVDESS codes map to correct labels."""
        from src.preprocess import get_emotion_from_filename

        assert get_emotion_from_filename("03-01-01-01-01-01-01.wav") == "neutral"
        assert get_emotion_from_filename("03-01-03-01-01-01-05.wav") == "happy"
        assert get_emotion_from_filename("03-01-05-01-01-01-12.wav") == "angry"
        assert get_emotion_from_filename("03-01-08-01-01-01-01.wav") == "surprised"

    def test_invalid_filename_raises(self):
        """A filename without a valid emotion code should raise."""
        from src.preprocess import get_emotion_from_filename

        with pytest.raises(ValueError):
            get_emotion_from_filename("badname.wav")
