import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Mnist_Model(BaseModel):
    def __init__(self, dropout_value=0.05):
        super(Mnist_Model, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
        )  # output_size = 6

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
        )  # output_size = 6

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))
        self.convblock8 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        )  # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Cifar10_Model(BaseModel):
    def __init__(self, dropout_rate=0.05):
        super(Cifar10_Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),  # In: 32x32x3 | Out: 32x32x32 | RF: 3x3
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # In: 32x32x32 | Out: 32x32x32 | RF: 5x5
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # In: 32x32x32 | Out: 16x16x32 | RF: 6x6
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),  # In: 16x16x32 | Out: 16x16x64 | RF: 10x10
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # In: 16x16x64 | Out: 16x16x64 | RF: 14x14
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # In: 16x16x64 | Out: 8x8x64 | RF:16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                groups=64,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),  # In: 8x8x64 | Out: 8x8x64 | RF: 24x24
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # In: 8x8x64 | Out: 8x8x64 | RF: 32x32
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # In: 8x8x64 | Out: 4x4x64 | RF: 36x36
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_rate),  # In: 4x4x64 | Out: 4x4x128 | RF: 68x68
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),  # In: 4x4x128 | Out: 4x4x128 | RF: 84x84
        )
        self.gap = nn.AdaptiveAvgPool2d(
            output_size=1
        )  # In: 4x4x128 | Out: 1x1x128 | RF: 108x108
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            # nn.ReLU() NEVER!
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.layer5(x)
        return x
