# -*- coding:utf-8 -*-
import os
import torch
import argparse
from classify import FaceRecognition
from torch.utils.data import DataLoader
from data_helper import FaceImageDataset, split_dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Train Parameter
    train = parser.add_argument_group('Train Option')
    train.add_argument('--img_path', type=str, default=os.path.join('.', 'data', 'image'))
    train.add_argument('--info_path', type=str, default=os.path.join('.', 'data', 'information.json'))
    train.add_argument('--split_rate', type=list, default=[0.8, 0.1, 0.1])

    train.add_argument('--epochs', type=int, default=80)
    train.add_argument('--batch_size', type=int, default=200)

    # Model
    vgg_network = parser.add_argument_group(title='VGG Network Option')
    vgg_network.add_argument('--vgg_type', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], default='vgg13')
    vgg_network.add_argument('--vgg_batch_norm', type=bool, default=True)

    inception_network = parser.add_argument_group('Google Inception Network Option')
    inception_network.add_argument('--inception_type', choices=['inception-v1'], default='inception-v1')

    return parser.parse_args()


class Trainer(object):
    def __init__(self, arguments):
        self.arguments = arguments
        self.dataset_info = split_dataset(arguments.info_path, split_rate=arguments.split_rate)

    def train(self):
        # DataLoader
        train_loader, val_loader = self.train_loader(), self.val_loader()

        # Training
        for epoch in range(self.arguments.epochs):
            for i, data in enumerate(train_loader):
                pass

    def train_loader(self) -> DataLoader:
        # train information List
        train_info = self.dataset_info[0]

        # train FaceImageDataset & DataLoader
        train_dataset = FaceImageDataset(
            root=self.arguments.img_path, info=train_info,
            mode='train', device=device
        )

        train_data_loader = DataLoader(
            dataset=train_dataset, batch_size=self.arguments.batch_size,
            shuffle=True
        )
        return train_data_loader

    def val_loader(self) -> DataLoader:
        # validation information.json List
        val_info = self.dataset_info[2]

        # validation FaceImageDataset & DataLoader
        val_dataset = FaceImageDataset(
            root=self.arguments.img_path, info=val_info,
            mode='validation', device=device
        )

        val_data_loader = DataLoader(
            dataset=val_dataset, batch_size=int(len(val_dataset) / 100)
        )
        return val_data_loader


if __name__ == '__main__':
    args = get_args()
    Trainer(args).train()
