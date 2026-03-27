import torch
import torch.nn as nn
from collections import OrderedDict


class AudioNet(nn.Module):
    """
    Lightweight waveform CNN (EnvNet-style)
    For event classification (ESC-50, UrbanSound, etc.)

    Input : (B, 1, T)
    Output: (B, num_classes)

    Key design:
    - Conv1D backbone
    - Global Average Pooling over time
    - Length-invariant (works for any T)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # --------------------------------------------------
        # Feature extractor
        # --------------------------------------------------
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38)),
            ("bn1",   nn.BatchNorm1d(32)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool1d(4)),

            ("conv2", nn.Conv1d(32, 64, kernel_size=3, padding=1)),
            ("bn2",   nn.BatchNorm1d(64)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool1d(4)),

            ("conv3", nn.Conv1d(64, 128, kernel_size=3, padding=1)),
            ("bn3",   nn.BatchNorm1d(128)),
            ("relu3", nn.ReLU(inplace=True)),
            ("pool3", nn.MaxPool1d(4)),
        ]))

        # --------------------------------------------------
        # Classifier
        # (after global temporal average → feature dim = 128)
        # --------------------------------------------------
        self.classifier = nn.Sequential(OrderedDict([
            ("fc1",   nn.Linear(128, 256)),
            ("relu4", nn.ReLU(inplace=True)),
            ("drop1", nn.Dropout(0.3)),
            ("fc2",   nn.Linear(256, num_classes)),
        ]))

    def forward(self, x):
        """
        x: (B, 1, T) or (B, 1, T, 1)
        """

        # normalize input shape
        if x.dim() == 4:          # (B,1,T,1)
            x = x.squeeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"AudioNet expects input shape (B,1,T), got {x.shape}")

        # Conv backbone
        x = self.features(x)      # (B, 128, T')

        # Global temporal average pooling
        x = x.mean(dim=-1)        # (B, 128)

        # Classifier
        x = self.classifier(x)    # (B, num_classes)

        return x
