"""PyTorch CNN–BiLSTM architecture for speech emotion classification."""

import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(40, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35),
        )
        self.lstm = nn.LSTM(256, 96, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.classifier(x.mean(dim=1))
