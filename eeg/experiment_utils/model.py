import math
import numpy as np
import torch
from torch import nn

# the implementations are a mixture of xbrainlab, matt and other sources
# (not word-by-word answer for BCI course homework, please do check the parameter settings)

class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        channels: int,
        samples: int,
        sfreq: float,
        F1: int = 8,
        F2: int = 16,
        D: int = 2
    ):
        super().__init__()

        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.half_sf = math.floor(self.sf/2)

        self.F1=F1
        self.F2=F2
        self.D=D

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1, self.F1, (1, self.half_sf), padding='valid', bias=False
            ),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                self.F1, self.D*self.F1, (self.ch, 1), groups=self.F1, bias=False
            ),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)), #reduce the sf to sf/4
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                self.D*self.F1,
                self.D*self.F1,
                (1, math.floor(self.half_sf/4)),
                padding='valid',
                groups=self.D*self.F1, bias=False
            ),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        fc_inSize = self._get_size(self.ch, self.tp)[1]
        self.classifier = nn.Linear(fc_inSize, self.n_class, bias=True)

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def _get_size(self, ch, tsamp):
        data = torch.ones((1, 1, ch, tsamp))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        return x.size()

class EEGNet_SSVEP(nn.Module):
    def __init__(
        self,
        n_classes: int,
        channels: int,
        samples: int,
        sfreq: float,
        F1: int = 100,
        F2: int = 10,
        D: int = 8
    ):
        super().__init__()

        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.half_sf = math.floor(self.sf/2)

        self.F1=F1
        self.F2=F2
        self.D=D

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1, self.F1, (1, self.half_sf), padding=(0, 100), bias=False
            ),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                self.F1, self.D*self.F1, (self.ch, 1), groups=self.F1, bias=False
            ),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)), #reduce the sf to sf/4
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                self.D*self.F1,
                self.D*self.F1,
                (1, 16),
                padding=(0,8),
                groups=self.D*self.F1, bias=False
            ),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        fc_inSize = self._get_size(self.ch, self.tp)[1]
        self.classifier = nn.Linear(fc_inSize, self.n_class, bias=True)

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def _get_size(self, ch, tsamp):
        data = torch.ones((1, 1, ch, tsamp))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        return x.size()


class SCCNet(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, Ns=22):
        super().__init__() 
        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.octsf = int(math.floor(self.sf*0.1))

        self.conv1 = nn.Conv2d(1, Ns, (self.ch, 1))
        self.Bn1 = nn.BatchNorm2d(Ns) #(n_ch)
        self.conv2 = nn.Conv2d(
            Ns, 20, (1, self.octsf), padding=(0, int(np.ceil((self.octsf-1)/2)))
        )
        self.Bn2   = nn.BatchNorm2d(20)

        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d(
            (1, int(self.sf/2)), stride=(1, int(self.octsf))
        )

        fc_inSize = self._get_size(self.ch, self.tp)[1]
        self.classifier = nn.Linear(fc_inSize, self.n_class, bias=True)

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        spX = self.conv1(x) #(128,22,1,562)

        x = self.Bn1(spX)
        tpX = self.conv2(x) #(128,20,1,563)

        x = self.Bn2(tpX)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x) #(128,20,1,42)
        x = torch.log(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        return x
    def _get_size(self, ch, tsamp):
        data = torch.ones((1, 1, ch, tsamp))
        x = self.conv1(data)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = self.AvgPool1(x)
        x = x.view(x.shape[0], -1)
        return x.size()

class InterpretableCNN(torch.nn.Module):  
    def __init__(self, n_classes=2, channels=30, samples=384, sfreq=128, N1=16, d=2):
        super(InterpretableCNN, self).__init__()
        self.pointwise = nn.Conv2d(1,N1,(channels,1))
        self.depthwise = nn.Conv2d(N1,d*N1,(1,math.floor(sfreq/2)), groups=N1) 
        self.activ=nn.ReLU()       
        self.batchnorm = nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP= nn.AvgPool2d((1, samples-math.floor(sfreq/2)+1))         
        self.fc = nn.Linear(d*N1, n_classes)        
        self.softmax= nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pointwise(x)        
        x = self.depthwise(x) 
        x = self.activ(x) 
        x = self.batchnorm(x)          
        x = self.GAP(x)     
        x = x.view(x.size()[0], -1) 
        x = self.fc(x)    
        x = self.softmax(x)   

        return x

class InterpretableCNN_SSVEP(torch.nn.Module):  
    def __init__(self, n_classes=2, channels=30, samples=384, sfreq=128, N1=100, d=8, kernelLength=64):
        super(InterpretableCNN_SSVEP, self).__init__()
        if kernelLength is None:
            kernelLength = -math.floor(sfreq/2)
        self.pointwise = nn.Conv2d(1,N1,(channels,1))
        self.depthwise = nn.Conv2d(N1,d*N1,(1,kernelLength), groups=N1) 
        self.activ=nn.ReLU()       
        self.batchnorm = nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP= nn.AvgPool2d((1, samples-kernelLength+1))         
        self.fc = nn.Linear(d*N1, n_classes)        
        self.softmax= nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pointwise(x)        
        x = self.depthwise(x) 
        x = self.activ(x) 
        x = self.batchnorm(x)          
        x = self.GAP(x)     
        x = x.view(x.size()[0], -1) 
        x = self.fc(x)    
        x = self.softmax(x)   

        return x
    







#______________________________________________________________________
class AlexNet_Audio(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        test_input = torch.zeros(1, 1, 227, 227)
        with torch.no_grad():
            f = self.features(test_input).shape  # (1, C, H, W)
        flattened = f[1] * f[2] * f[3]

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flattened, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class AudioNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # conv1d expects (batch, channels, time)
        # original data shape: (B,1,1,8000)-> reshape to (B,1,8000)

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # auto-calc linear input dim
        test_input = torch.zeros(1, 1, 8000)
        with torch.no_grad():
            out = self.features(test_input).shape
        flattened = out[1] * out[2]

        self.classifier = nn.Sequential(
            nn.Linear(flattened, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 1, 8000)
        x = x.squeeze(2)  # → (B,1,8000)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x