#input is waveform in training script, but actually process spectrogram 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation



def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, pool_size)
        return x

class Cnn14(nn.Module):
    """
    CNN14 (waveform version) for ESC-50

    Input:
        x: (B, T)

    Output:
        logits: (B, 50)
    """

    def __init__(
        self,
        sample_rate=8000,
        window_size=1024,
        hop_size=256,
        mel_bins=64,
        fmin=50,
        fmax=4000,
        num_classes=50,
    ):
        super().__init__()

        # -------- frontend --------
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        # -------- CNN14 backbone --------
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)


        self.fc1 = nn.Linear(2048, 2048)
        self.fc_out = nn.Linear(2048, num_classes)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_out)

    def forward(self, x):
        """
        x: (B, T)
        """

        # ---- waveform -> log-mel ----
        x = self.spectrogram_extractor(x)     # (B,1,time,freq)
        x = self.logmel_extractor(x)           # (B,1,time,mel)

        x = x.transpose(1, 3)                  # (B,mel,time,1)
        x = self.bn0(x)
        x = x.transpose(1, 3)                  # (B,1,time,mel)

        if self.training and self.spec_augmenter is not None:
            x = self.spec_augmenter(x)

        # ---- CNN backbone ----
        x = self.conv_block1(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block5(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, 0.2, self.training)

        # ---- global pooling ----
        x = torch.mean(x, dim=3)       # freq
        x1, _ = torch.max(x, dim=2)    # time max
        x2 = torch.mean(x, dim=2)      # time avg
        x = x1 + x2

        x = F.dropout(x, 0.5, self.training)
        x = F.relu_(self.fc1(x))
        logits = self.fc_out(x)

        return logits
    



def load_cnn14_pretrained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"]


    for k in ["fc_audioset.weight", "fc_audioset.bias"]:
        if k in state_dict:
            state_dict.pop(k)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[CNN14 waveform] pretrained loaded")
    print("  Missing:", missing)
    print("  Unexpected:", unexpected)
