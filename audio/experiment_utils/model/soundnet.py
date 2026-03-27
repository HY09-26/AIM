import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundNet8(nn.Module):
    def __init__(self):
        super().__init__()

        # input: (B, 1, T, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # conv8: representation layer (often no BN in some variants)
        self.conv8 = nn.Conv2d(
            1024, 1000, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: (B, 1, T) → (B, 1, T, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool(x)

        return x.view(x.size(0), -1)  # (B, 1000)
