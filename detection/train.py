# -*- coding:utf-8 -*-
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from detection.data_helper import FaceImageDataset, split_dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Train Parameter
    train = parser.add_argument_group('Train Option')
    train.add_argument('--img_path', type=str, default=os.path.join('.', 'data', 'image'))
    train.add_argument('--info_path', type=str, default=os.path.join('.', 'data', 'information.json'))
    train.add_argument('--split_rate', type=list, default=[0.8, 0.1, 0.1])

    train.add_argument('--epochs', type=int, default=80)
    train.add_argument('--batch_size', type=int, default=500)
    train.add_argument('--model', choices=['vgg', 'inception'], type=str, default='vgg')
    train.add_argument('--learning_rate', type=float, default=0.01)
    train.add_argument('--print_train_step', type=int, default=1)
    train.add_argument('--print_val_step', type=int, default=200)
    train.add_argument('--saving_point_step', type=int, default=100)

    # Model
    vgg_network = parser.add_argument_group(title='VGG Network Option')
    vgg_network.add_argument('--vgg_type', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], type=str, default='vgg13')
    vgg_network.add_argument('--vgg_batch_norm', type=bool, default=True)

    inception_network = parser.add_argument_group('Google Inception Network Option')
    inception_network.add_argument('--inception_type', choices=['inception_v1'], default='inception-v1')

    return parser.parse_args()


class Trainer(object):
    def __init__(self, arguments):
        self.arguments = arguments
        self.dataset_info = split_dataset(arguments.info_path, split_rate=arguments.split_rate)
        self.model = self.get_model()       # Define FaceRecognition
        self.sex_branch_criterion = nn.CrossEntropyLoss()
        self.age_branch_criterion = nn.CrossEntropyLoss()
        self.optimizer = opt.SGD(
            self.model.parameters(),
            lr=self.arguments.learning_rate
        )

    def train(self):
        # Train DataLoader
        train_loader = self.train_loader()

        # model & optimizer & loss
        self.model.to(device)
        self.model.train()

        # Training
        for epoch in range(self.arguments.epochs):
            for i, data in enumerate(train_loader):
                out = self.model(data['img'])
                del data['img']

                out = self.get_accuracy_loss(out=out, tar=data)
                loss = out['loss']['sex'] + out['loss']['age']

                if i % self.arguments.print_train_step == 0:
                    self.print_fn(mode='Train',
                                  epoch=epoch, it=i,
                                  total_loss=loss,
                                  sex_loss=out['loss']['sex'].item(),
                                  age_loss=out['loss']['age'].item(),
                                  sex_accuracy=out['accuracy']['sex'].item(),
                                  age_accuracy=out['accuracy']['age'].item())

                if i % self.arguments.print_val_step == 0:
                    val_total_loss, val_total_sex_loss, val_total_age_loss, val_total_sex_acc, val_total_age_acc = \
                        self.val()
                    self.print_fn(mode=' Val ',
                                  epoch=epoch, it=i,
                                  total_loss=val_total_loss,
                                  sex_loss=val_total_sex_loss, age_loss=val_total_age_loss,
                                  sex_accuracy=val_total_sex_acc, age_accuracy=val_total_age_acc)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def val(self):
        # Validation DataLoader
        val_loader = self.val_loader()
        total_loss = 0
        total_sex_loss, total_age_loss = 0, 0
        total_sex_acc, total_age_acc = 0, 0

        count = 0
        with torch.no_grad():
            for data in val_loader:
                out = self.model(data['img'])
                del data['img']

                out = self.get_accuracy_loss(out=out, tar=data)
                sex_loss, age_loss = out['loss']['sex'].item(), out['loss']['age'].item()
                # calculation loss & accuracy
                total_loss += sex_loss + age_loss
                total_sex_loss += sex_loss
                total_age_loss += age_loss
                total_sex_acc += out['accuracy']['sex'].item()
                total_age_acc += out['accuracy']['age'].item()
                count += 1

        # calculation average loss & accuracy
        total_loss /= count
        total_sex_loss /= count
        total_age_loss /= count
        total_age_acc /= count
        total_sex_acc /= count
        return total_loss, total_sex_loss, total_age_loss, total_sex_acc, total_age_acc

    def get_accuracy_loss(self, out, tar):
        sex_out, age_out = out['sex'].to(device), out['age'].to(device)
        sex_tar, age_tar = tar['sex'].to(device), tar['age'].to(device)

        # calculation loss
        sex_loss = self.sex_branch_criterion(
            input=sex_out,
            target=sex_tar
        )
        age_loss = self.age_branch_criterion(
            input=age_out,
            target=age_tar
        )

        # calculation accuracy
        sex_indicate = sex_out.argmax(dim=-1)
        sex_accuracy = sex_tar.eq(sex_indicate).sum().to(torch.float32) / sex_indicate.size()[0]
        age_indicate = age_out.argmax(dim=-1)
        age_accuracy = age_tar.eq(age_indicate).sum().to(torch.float32) / age_indicate.size()[0]

        return {
            'loss': {'sex': sex_loss, 'age': age_loss},
            'accuracy': {'sex': sex_accuracy, 'age': age_accuracy}
        }

    def train_loader(self) -> DataLoader:
        # train information.json List
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

    def get_model(self):
        parameter = self.get_parameter()
        model_name = self.arguments.model
        if model_name == 'vgg':
            from detection.models.vgg import VGG
            return VGG(**parameter)
        elif model_name == 'inception':
            from detection.models.inception import InceptionV1
            return

    def get_parameter(self):
        model_name = self.arguments.model
        parameter = {}
        if model_name == 'vgg':
            parameter = {
                'vgg_type': self.arguments.vgg_type,
                'batch_norm': self.arguments.vgg_batch_norm
            }
        elif model_name == 'inception':
            parameter = {
                'type': self.arguments.inception_type
            }
        return parameter

    @staticmethod
    def print_fn(epoch, it, total_loss, sex_loss, age_loss, sex_accuracy, age_accuracy, mode='Train'):
        print(
            '[ {0} ] => '
            '[Epoch] : {1:3d}  [Iter] : {2:3d}  [Total Loss] : {3:9.4f}  '
            '[Sex Loss] : {4:9.4f}  [Sex Accuracy] : {6:9.4f}  '
            '[Age Loss] : {5:9.4f}  [Age Accuracy] : {7:9.4f}'.format(
                    mode,
                    epoch, it, total_loss,
                    sex_loss, age_loss,
                    sex_accuracy, age_accuracy)
        )


if __name__ == '__main__':
    args = get_args()
    Trainer(args).train()
