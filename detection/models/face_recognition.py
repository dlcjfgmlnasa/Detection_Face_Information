# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from detection.models.classify import Classify


class BasicCNN(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=output_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.layer(inputs)
        return x


class NaiveInceptionV1(nn.Module):
    def __init__(self, in_channels, _1x1, _3x3, _5x5, _pool_proj):
        super().__init__()
        self.branch1 = BasicCNN(in_channels=in_channels, output_channels=_1x1, kernel_size=(1, 1), stride=1)
        self.branch2 = BasicCNN(in_channels=in_channels, output_channels=_3x3, kernel_size=(3, 3), stride=1, padding=1)
        self.branch3 = BasicCNN(in_channels=in_channels, output_channels=_5x5, kernel_size=(5, 5), stride=1, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            BasicCNN(in_channels=in_channels, output_channels=_pool_proj, kernel_size=(1, 1), stride=1)
        )

    def forward(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        x4 = self.branch4(inputs)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class FRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem layer
        self.stem = nn.Sequential(
            BasicCNN(in_channels=3, output_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # InceptionV1 Convolution Layer
        self.inception = nn.Sequential(
            NaiveInceptionV1(
                in_channels=32, _1x1=32, _3x3=32, _5x5=16, _pool_proj=16
            ),
            NaiveInceptionV1(
                in_channels=96, _1x1=32, _3x3=32, _5x5=16, _pool_proj=16
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected layer
        self.fc = Classify(input_size=24576)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



