"""Predict emotion from an audio clip using the notebook-trained PyTorch model."""

import os
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import extract_mfcc_sequence_from_file
from src.model import CNNBiLSTM

_model = None
_mean = None
_std = None
_labels = None
_loaded_model_dir = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_artefacts(model_dir: str = "data/models") -> None:
    global _model, _mean, _std, _labels, _loaded_model_dir

    model_path = os.path.join(model_dir, "best_ravdess_pytorch.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PyTorch checkpoint not found at '{model_path}'. "
            "Train the model or copy the notebook checkpoint into data/models."
        )
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    _labels = checkpoint["classes"].tolist()
    _model = CNNBiLSTM(len(_labels)).to(_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    _mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(-1)
    _std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(-1)
    _loaded_model_dir = os.path.abspath(model_dir)


def predict_emotion(file_path: str, model_dir: str = "data/models") -> dict:
    """Return predicted emotion, confidence, and probabilities for one audio file."""
    global _model, _mean, _std, _labels, _loaded_model_dir
    if _model is None or _loaded_model_dir != os.path.abspath(model_dir):
        _load_artefacts(model_dir)

    try:
        features = extract_mfcc_sequence_from_file(file_path)
    except Exception as exc:
        raise ValueError(f"Could not process audio file '{file_path}': {exc}") from exc

    normalized = (features - _mean) / _std
    model_input = torch.as_tensor(normalized, dtype=torch.float32)
    model_input = model_input.permute(1, 0).unsqueeze(0).to(_device)
    with torch.no_grad():
        probabilities = torch.softmax(_model(model_input), dim=1)[0].cpu().numpy()
    predicted_idx = int(np.argmax(probabilities))
    return {
        "emotion": _labels[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(_labels, probabilities)
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict emotion from a speech audio file.")
    parser.add_argument("audio_file", type=str)
    parser.add_argument("--model-dir", type=str, default="data/models")
    args = parser.parse_args()
    result = predict_emotion(args.audio_file, args.model_dir)
    print(f"Predicted emotion: {result['emotion']} ({result['confidence']:.2%})")
    for label, probability in sorted(result["probabilities"].items(),
                                     key=lambda item: item[1], reverse=True):
        print(f"  {label:<12} {probability:.2%}")
