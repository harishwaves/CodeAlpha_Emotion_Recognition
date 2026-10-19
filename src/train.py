"""Train the PyTorch CNN–BiLSTM with speaker-independent evaluation."""

import os
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import CNNBiLSTM
from src.preprocess import EMOTION_LABELS
from src.splits import actor_independent_split

SEED = 42
EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 10
WEIGHT_DECAY = 1e-3
NOISE_STD_MAX = 0.08
TIME_MASK_MAX = 27
FREQ_MASK_MAX = 5


def _make_loader(features, targets, device, shuffle=False):
    x_tensor = torch.as_tensor(features, dtype=torch.float32).permute(0, 2, 1)
    y_tensor = torch.as_tensor(targets, dtype=torch.long)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
    )


def _augment_mfcc_batch(x):
    """Apply SpecAugment-style masking/noise to normalized training MFCCs."""
    batch_size, n_coefficients, n_frames = x.shape
    noise_scale = torch.empty(
        (batch_size, 1, 1), device=x.device, dtype=x.dtype
    ).uniform_(0.0, NOISE_STD_MAX)
    augmented = x + torch.randn_like(x) * noise_scale

    time_width = torch.randint(
        6, TIME_MASK_MAX + 1, (batch_size,), device=x.device
    )
    time_start = torch.floor(
        torch.rand(batch_size, device=x.device)
        * (n_frames - TIME_MASK_MAX + 1)
    ).long()
    frame_indices = torch.arange(n_frames, device=x.device).unsqueeze(0)
    time_mask = (frame_indices >= time_start.unsqueeze(1)) & (
        frame_indices < (time_start + time_width).unsqueeze(1)
    )
    augmented = augmented.masked_fill(time_mask.unsqueeze(1), 0.0)

    coefficient_width = torch.randint(
        1, FREQ_MASK_MAX + 1, (batch_size,), device=x.device
    )
    coefficient_start = torch.floor(
        torch.rand(batch_size, device=x.device)
        * (n_coefficients - FREQ_MASK_MAX + 1)
    ).long()
    coefficient_indices = torch.arange(n_coefficients, device=x.device).unsqueeze(0)
    coefficient_mask = (coefficient_indices >= coefficient_start.unsqueeze(1)) & (
        coefficient_indices < (coefficient_start + coefficient_width).unsqueeze(1)
    )
    return augmented.masked_fill(coefficient_mask.unsqueeze(2), 0.0)


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    true, predicted = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * len(yb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
            true.extend(yb.cpu().numpy())
            predicted.extend(pred.cpu().numpy())
    return loss_sum / total, correct / total, np.asarray(true), np.asarray(predicted)


def train(data_dir: str = "data", model_dir: str = "data/models") -> None:
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    groups_path = os.path.join(data_dir, "groups.npy")
    if not all(os.path.exists(path) for path in (x_path, y_path, groups_path)):
        raise FileNotFoundError(
            f"Processed X.npy, y.npy, and groups.npy not found in '{data_dir}'. "
            "Run src.preprocess first to generate features and actor IDs."
        )
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    groups = np.load(groups_path)
    if (X.shape[1:] != (216, 40) or y.ndim != 1 or len(X) != len(y)
            or groups.ndim != 1 or len(groups) != len(y)):
        raise ValueError(
            "Expected X shaped (samples, 216, 40) plus matching one-dimensional "
            "y and groups arrays. Re-run preprocessing using the notebook MFCC pipeline."
        )
    if len(np.unique(y)) != len(EMOTION_LABELS):
        raise ValueError("Training data must contain all eight RAVDESS emotions, including calm.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    (train_indices, val_indices, test_indices,
     train_actors, val_actors, test_actors) = actor_independent_split(groups, SEED)
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    for split_name, split_labels in (
        ("train", y_train), ("validation", y_val), ("test", y_test)
    ):
        if len(np.unique(split_labels)) != len(EMOTION_LABELS):
            raise ValueError(
                f"The actor-independent {split_name} split is missing an emotion class. "
                "Check data/groups.npy or rerun preprocessing."
            )
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train, X_val, X_test = ((part - mean) / std
                              for part in (X_train, X_val, X_test))
    print(
        f"[INFO] Actor-independent split: train={len(y_train)} clips/"
        f"{len(train_actors)} actors, validation={len(y_val)} clips/"
        f"{len(val_actors)} actors, test={len(y_test)} clips/"
        f"{len(test_actors)} actors"
    )

    train_loader = _make_loader(X_train, y_train, device, shuffle=True)
    train_eval_loader = _make_loader(X_train, y_train, device)
    val_loader = _make_loader(X_val, y_val, device)
    test_loader = _make_loader(X_test, y_test, device)

    model = CNNBiLSTM(len(EMOTION_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_ravdess_pytorch.pt")
    best_loss, stale = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = _augment_mfcc_batch(xb)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Measure training accuracy on clean examples so it is comparable to validation.
        _, train_accuracy, _, _ = _evaluate(
            model, train_eval_loader, criterion, device
        )
        val_loss, val_accuracy, _, _ = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:02d}: train acc {train_accuracy:.3f} | "
            f"val acc {val_accuracy:.3f} | val loss {val_loss:.4f}"
        )
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": np.asarray(EMOTION_LABELS),
                "mean": mean,
                "std": std,
                "split_method": "actor-independent",
                "train_actors": train_actors,
                "validation_actors": val_actors,
                "test_actors": test_actors,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("[INFO] Early stopping.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy, _, _ = _evaluate(model, test_loader, criterion, device)
    np.save(os.path.join(model_dir, "label_encoder.npy"), np.asarray(EMOTION_LABELS))
    np.save(os.path.join(model_dir, "X_test.npy"), X_test)
    np.save(os.path.join(model_dir, "y_test.npy"), y_test)
    print(f"[TEST] Loss: {test_loss:.4f}")
    print(f"[TEST] Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"[DONE] Saved PyTorch CNN–BiLSTM checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the PyTorch CNN–BiLSTM model.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="data/models")
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
