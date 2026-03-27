import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvNet(nn.Module):
    """
    EnvNet (Tokozume & Harada, 2017)
    Raw waveform -> 1D conv -> reshape -> 2D conv
    Input: (B, 1, 24000)  # 1.5 sec @ 16 kHz
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ===== Stage 1: 1D temporal convolution =====
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=40,
            kernel_size=80,
            stride=4,
            padding=38
        )
        self.bn1 = nn.BatchNorm1d(40)

        self.conv2 = nn.Conv1d(
            in_channels=40,
            out_channels=40,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(40)

        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # ===== Stage 2: reshape to 2D =====
        # (B, 40, T) -> (B, 1, 40, T)

        # ===== Stage 3: 2D convolution on feature map =====
        self.conv3 = nn.Conv2d(1, 50, kernel_size=(8, 13))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 5))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv5 = nn.Conv2d(50, 50, kernel_size=(1, 5))
        self.bn5 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # ===== classifier =====
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 24000)
            x = self._forward_features(dummy)
            feat_dim = x.shape[1]

        self.fc1 = nn.Linear(feat_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.drop = nn.Dropout(0.5)

    def _forward_features(self, x):
        # x: (B, 1, 24000)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.pool1(x)                 # (B, 40, T')

        # reshape to 2D
        x = x.unsqueeze(1)                # (B, 1, 40, T')

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool2(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool3(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool4(x)

        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self._forward_features(x)

        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
