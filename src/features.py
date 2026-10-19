"""
features.py — Audio Feature Extraction Module
===============================================
Extracts four feature groups from audio signals:
  • MFCCs  (Mel-Frequency Cepstral Coefficients)
  • Chroma (pitch-class energy distribution)
  • Spectral Contrast
  • Zero-Crossing Rate

All features are averaged over time and concatenated into a single
fixed-length vector suitable for classification.
"""

import numpy as np
import librosa

# ── Configuration ──────────────────────────────────────────────────────────
SAMPLE_RATE = 22050       # Librosa default sample-rate
DURATION = 3.0            # Seconds – clips/pads every file to this length
N_MFCC = 40               # Number of MFCC coefficients
N_CHROMA = 12              # Chroma bins (one per pitch class)
N_CONTRAST = 7             # Spectral-contrast sub-bands
N_SEGMENTS = 6              # Preserve coarse emotion changes over a 3-second clip
N_MFCC = 40
MAX_MFCC_FRAMES = 216


def load_audio(file_path: str,
               sr: int = SAMPLE_RATE,
               duration: float = DURATION) -> np.ndarray:
    """
    Load an audio file, resample to ``sr``, and pad/truncate to
    exactly ``duration`` seconds.

    Parameters
    ----------
    file_path : str
        Path to a .wav or .mp3 file.
    sr : int
        Target sample rate.
    duration : float
        Target duration in seconds.

    Returns
    -------
    np.ndarray
        1-D waveform array of length ``sr * duration``.
    """
    # Load audio (mono, resampled)
    y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)

    # Pad with zeros if shorter than target length
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]

    return y


def extract_features(y: np.ndarray,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract and concatenate audio features from a waveform.

    Features (all time-averaged):
        • 40  MFCCs
        • 12  Chroma values
        •  7  Spectral-contrast bands
        •  1  Zero-crossing rate  (mean)

    Parameters
    ----------
    y : np.ndarray
        1-D audio waveform.
    sr : int
        Sample rate of ``y``.

    Returns
    -------
    np.ndarray
        1-D feature vector of length 60.
    """
    # ── MFCCs ──────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs, axis=1)                       # (40,)

    # ── Chroma ─────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    chroma_mean = np.mean(chroma, axis=1)                      # (12,)

    # ── Spectral Contrast ──────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                  n_bands=N_CONTRAST - 1)
    contrast_mean = np.mean(contrast, axis=1)                  # (7,)

    # ── Zero-Crossing Rate ─────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)                                    # scalar

    # ── Concatenate all features ───────────────────────────────────────
    feature_vector = np.concatenate([
        mfccs_mean,
        chroma_mean,
        contrast_mean,
        [zcr_mean],
    ])

    return feature_vector  # shape: (60,)


def extract_features_from_file(file_path: str) -> np.ndarray:
    """
    Convenience wrapper: load an audio file and return its feature vector.

    Parameters
    ----------
    file_path : str
        Path to a .wav or .mp3 file.

    Returns
    -------
    np.ndarray
        1-D feature vector of length 60.

    Raises
    ------
    Exception
        If the file cannot be loaded or processed.
    """
    y = load_audio(file_path)
    return extract_features(y)


def extract_feature_sequence_from_file(file_path: str,
                                      n_segments: int = N_SEGMENTS) -> np.ndarray:
    """Return one 60-D feature vector per equal-duration audio segment."""
    if n_segments < 1:
        raise ValueError("n_segments must be positive")
    y = load_audio(file_path)
    segments = np.array_split(y, n_segments)
    return np.stack([extract_features(segment) for segment in segments]).astype(np.float32)


def extract_mfcc_sequence_from_file(file_path: str,
                                    sr: int = SAMPLE_RATE,
                                    max_frames: int = MAX_MFCC_FRAMES) -> np.ndarray:
    """Extract the notebook's fixed-length (time, 40 MFCC) representation."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    frames = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    if len(frames) < max_frames:
        frames = np.pad(frames, ((0, max_frames - len(frames)), (0, 0)), mode="constant")
    return np.asarray(frames[:max_frames], dtype=np.float32)
