# -*- coding:utf -*-
import torch.nn as nn
import torch.nn.functional as f


class SexClassify(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class AgeClassify(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
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
        sex_out = f.softmax(sex_out, dim=-1)
        age_out = f.softmax(age_out, dim=-1)
        return {'sex': sex_out, 'age': age_out}
