# -*- coding:utf-8 -*-
import os
import json
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

img_size = 64


class FaceImageDataset(Dataset):
    def __init__(self, root, info, device, mode='train'):
        self.root = root
        self.info = info
        self.mode = mode
        self.device = device

        if mode not in ['train', 'validation', 'test']:
            raise NotImplemented('You have to choose one here [train, test, validation]')

    def __getitem__(self, idx):
        temp = self.info[idx]
        img_path = os.path.join(self.root, temp['image'])

        img = self.image_preprocess(img_path)
        sex = (0 if temp['sex'] == 'male' else 1)
        age = self._age_categorization(temp['age'])

        return {
            'img': img,
            'sex': sex,
            'age': age
        }

    def image_preprocess(self, img_path):
        img = Image.open(img_path)
        if self.mode == 'train':
            preprocess = transforms.Compose([transforms.Resize(img_size+4),
                                             transforms.RandomCrop(img_size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
            img = preprocess(img)

        elif self.mode == 'validation':
            preprocess = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
            img = preprocess(img)

        elif self.mode == 'test':
            preprocess = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
            img = preprocess(img)

        return img.to(self.device)

    @staticmethod
    def _age_categorization(age):
        age = age / 10
        return int(age)

    def __len__(self):
        return len(self.info)


def split_dataset(path, split_rate: list):
    with open(path, 'r') as fp:
        dataset = json.load(fp)
    random.shuffle(dataset)
    size = len(dataset)

    train_set = dataset[0: int(size*split_rate[0])]
    test_set = dataset[int(size*split_rate[0]): int(size*(split_rate[0] + split_rate[1]))]
    validation_set = dataset[int(size*(split_rate[0] + split_rate[1])):]
    return train_set, test_set, validation_set


def dataset_info(path):
    with open(path, 'r') as fp:
        dataset = json.load(fp)
    df = pd.DataFrame(dataset)
    df['age'] = (df['age'] / 10).astype(int)
    age_temp, sex_temp = [], []
    age_group, sex_group = df.groupby('age'), df.groupby('sex')
    for age, group in age_group.groups.items():
        age_temp.append({
            'age': '{}대'.format(age * 10),
            'count': len(group)
        })
        
    for sex, group in sex_group.groups.items():
        sex_temp.append({
            'sex': sex,
            'count': len(group)
        })
    age_df = pd.DataFrame(age_temp)
    sex_df = pd.DataFrame(sex_temp)
    return {'age': age_df, 'sex': sex_df}
