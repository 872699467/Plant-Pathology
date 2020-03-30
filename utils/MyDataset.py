import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from PIL import Image


def get_train_trans(SIZE):
    return transforms.Compose([
        transforms.CenterCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[0.3, 0.2, 0.1])
    ])


def get_test_trans(SIZE):
    return transforms.Compose([
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[0.3, 0.2, 0.1])
    ])


class TrainDataSet(Dataset):

    def __init__(self, base_path, img_list, target_list, transformation=None):
        super(TrainDataSet, self).__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.target_list = target_list
        self.trans = transformation

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.base_path, self.img_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img = self.trans(img)
        target = torch.tensor(self.target_list[index])
        return img, target

    def collate_fn(self, item):
        imgs, targets = list(zip(*item))
        imgs = torch.stack(imgs, dim=0)
        targets = torch.stack(targets, dim=0)
        return imgs, targets


class TestDataset(Dataset):

    def __init__(self, base_path, img_list, transformation=None):
        super(TestDataset, self).__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.trans = transformation

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.base_path, self.img_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img = self.trans(img)
        return img

    def collate_fn(self, item):
        imgs = item
        imgs = torch.stack(imgs, dim=0)
        return imgs
