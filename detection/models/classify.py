# -*- coding:utf -*-
import torch.nn as nn
import torch.nn.functional as f


class Output(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        self.input_size = input_size
        self.sex_branch = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2)
        )
        self.age_branch = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 100)
        )

    def forward(self, inputs):
        sex_out = self.sex_branch(inputs)
        age_out = self.age_branch(inputs)
        sex_out = f.log_softmax(sex_out, dim=-1)
        age_out = f.log_softmax(age_out, dim=-1)
        return {'sex': sex_out, 'age': age_out}
