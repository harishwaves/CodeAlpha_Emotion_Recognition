"""
preprocess.py — Dataset Preparation for RAVDESS
=================================================
Walks the RAVDESS dataset directory, extracts features from every
audio file, maps each file to its emotion label, and saves the
processed data as NumPy arrays for training.

RAVDESS filename convention (Actor_XX/03-01-{EMOTION}-01-01-01-XX.wav):
    Emotion codes:  01=neutral, 02=calm, 03=happy, 04=sad,
                    05=angry, 06=fearful, 07=disgust, 08=surprised

We keep all eight RAVDESS emotion labels, including both neutral and calm.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# Allow imports when running as a script from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import extract_mfcc_sequence_from_file

# ── Emotion Label Mapping ─────────────────────────────────────────────────
# RAVDESS emotion codes → human-readable labels
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Alphabetical ordering matches scikit-learn's LabelEncoder in the notebook.
EMOTION_LABELS = sorted(set(RAVDESS_EMOTION_MAP.values()))
#  ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}


def get_emotion_from_filename(filename: str) -> str:
    """
    Parse the RAVDESS filename and return the emotion label.

    Parameters
    ----------
    filename : str
        Filename like ``03-01-05-01-01-01-12.wav``.

    Returns
    -------
    str
        Emotion label (e.g. ``'angry'``).

    Raises
    ------
    ValueError
        If the emotion code is unrecognised.
    """
    # The emotion code is the 3rd part when split by "-"
    parts = os.path.splitext(filename)[0].split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    code = parts[2]
    if code not in RAVDESS_EMOTION_MAP:
        raise ValueError(f"Unknown emotion code '{code}' in {filename}")

    return RAVDESS_EMOTION_MAP[code]


def preprocess_dataset(data_dir: str, output_dir: str = "data") -> None:
    """
    Walk the RAVDESS dataset, extract features, and save as NumPy files.

    Expected directory layout::

        data_dir/
        ├── Actor_01/
        │   ├── 03-01-01-01-01-01-01.wav
        │   └── ...
        ├── Actor_02/
        └── ...

    Outputs saved to ``output_dir/``:
        - ``X.npy``  — MFCC sequences (N, 216, 40)
        - ``y.npy``  — integer labels   (N,)

    Parameters
    ----------
    data_dir : str
        Root path containing Actor_XX subdirectories.
    output_dir : str
        Directory to save processed .npy files.
    """
    features_list = []
    labels_list = []
    groups_list = []
    skipped = 0

    # Collect all audio file paths first
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for fname in sorted(files):
            if fname.lower().endswith((".wav", ".mp3")):
                audio_files.append(os.path.join(root, fname))

    if not audio_files:
        print(f"[ERROR] No audio files found in '{data_dir}'.")
        print("        Please download the RAVDESS dataset first — see README.md.")
        sys.exit(1)

    print(f"[INFO] Found {len(audio_files)} audio files. Extracting features …")

    for filepath in tqdm(audio_files, desc="Processing", unit="file"):
        fname = os.path.basename(filepath)
        try:
            emotion = get_emotion_from_filename(fname)
            feat = extract_mfcc_sequence_from_file(filepath)
            features_list.append(feat)
            labels_list.append(LABEL_TO_INDEX[emotion])
            # The final RAVDESS filename field is the speaker (actor) ID.
            groups_list.append(int(os.path.splitext(fname)[0].split("-")[-1]))
        except Exception as exc:
            skipped += 1
            tqdm.write(f"  ⚠ Skipped {fname}: {exc}")

    if not features_list:
        raise RuntimeError(
            "No files could be processed. Check that the dataset contains "
            "valid RAVDESS audio filenames and that the audio dependencies "
            "are installed."
        )

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    groups = np.array(groups_list, dtype=np.int64)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    np.save(os.path.join(output_dir, "groups.npy"), groups)

    print(f"\n[DONE] Saved {X.shape[0]} samples → {output_dir}/X.npy, y.npy, groups.npy")
    print(f"       Feature shape : {X.shape} (samples, MFCC frames, coefficients)")
    print(f"       Classes       : {EMOTION_LABELS}")
    if skipped:
        print(f"       Skipped       : {skipped} files")


# ── CLI Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-process the RAVDESS dataset for emotion recognition."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Audio_Speech_Actors_01-24",
        help=("Path to RAVDESS root directory "
              "(default: data/Audio_Speech_Actors_01-24)."),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save X.npy and y.npy (default: data).",
    )
    args = parser.parse_args()
    preprocess_dataset(args.data_dir, args.output_dir)
