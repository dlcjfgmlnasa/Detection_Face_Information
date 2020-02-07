# -*- coding:utf -*-
import torch.nn as nn
import torch.nn.functional as f


class SexClassify(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 2)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class AgeClassify(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 10)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class Classify(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        self.sex_classify = SexClassify(input_size)
        self.age_classify = AgeClassify(input_size)

    def forward(self, inputs):
        sex_out = self.sex_classify(inputs)
        age_out = self.age_classify(inputs)
        return {'sex': sex_out, 'age': age_out}
