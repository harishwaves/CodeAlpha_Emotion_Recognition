"""Evaluate the saved PyTorch model on the actor-independent test split."""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import librosa
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import CNNBiLSTM
from src.splits import actor_independent_split


def evaluate(data_dir: str = "data", model_dir: str = "data/models",
             output_dir: str = "data/models") -> None:
    model_path = os.path.join(model_dir, "best_ravdess_pytorch.pt")
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    groups_path = os.path.join(data_dir, "groups.npy")
    for path in (model_path, x_path, y_path, groups_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required evaluation file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if checkpoint.get("split_method") != "actor-independent":
        raise ValueError(
            "This checkpoint was not trained with an actor-independent split. "
            "Run src.train first so evaluation uses actors held out from training."
        )
    classes = checkpoint["classes"].tolist()
    model = CNNBiLSTM(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X, y = np.load(x_path).astype(np.float32), np.load(y_path).astype(np.int64)
    groups = np.load(groups_path)
    if X.shape[1:] != (216, 40) or len(X) != len(y) or len(groups) != len(y):
        raise ValueError(
            "Expected matching MFCC, label, and actor arrays shaped "
            "(samples, 216, 40), (samples,), and (samples,); rerun preprocessing."
        )
    _, _, test_indices, _, _, test_actors = actor_independent_split(groups, seed=42)
    X_test, y_test = X[test_indices], y[test_indices]
    X_test = (X_test - checkpoint["mean"]) / checkpoint["std"]
    inputs = torch.as_tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        predicted = model(inputs).argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predicted)
    print(f"[TEST] Actor-independent test actors: {test_actors.tolist()}")
    print(f"[TEST] Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(classification_report(y_test, predicted, target_names=classes, zero_division=0))

    os.makedirs(output_dir, exist_ok=True)
    matrix = confusion_matrix(y_test, predicted, labels=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("RAVDESS Emotion Classification — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the PyTorch emotion model.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="data/models")
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_dir)
