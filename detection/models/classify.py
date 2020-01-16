# -*- coding:utf -*-
import torch.nn as nn
from detection.models import *


class Output(nn.Module):
    def __init__(self, input_size=512):
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
            nn.Linear(4096, 1)
        )

    def forward(self, inputs):
        sex_out = self.sex_branch(inputs)
        age_out = self.age_branch(inputs)
        return sex_out, age_out


class FaceRecognition(object):
    def __init__(self, model, parameter, device):
        self.model = model
        self.parameter = parameter
        self.device = device
        self.sex_branch_criterion = nn.CrossEntropyLoss()
        self.age_branch_criterion = nn.MSELoss()

    def get_model(self):
        model = None
        if self.model == 'vgg':
            from detection.models.vgg import VGG
            model = VGG(
                vgg_type=self.parameter['type'],
                batch_norm=self.parameter['batch_norm']
            )
        elif self.model == 'inception':
            pass
        return model.to(self.device)

    def detection(self, tar):
        pass
