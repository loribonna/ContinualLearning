# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Iterator
from torch.utils.data import DataLoader


class Sequentializer:
    def __init__(self, data_builder=None, train=None, test=None, batch_size=64, transforms=[torchvision.transforms.ToTensor()], no_cuda=False):
        self.data_builder = data_builder
        self.train = train
        self.test = test
        self.transforms = transforms
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self._setup()

    def _setup(self):
        if (self.data_builder):
            self.train = self.data_builder('/data/',
                                           train=True, download=True,
                                           transform=torchvision.transforms.Compose(
                                               self.transforms)
                                           )
            self.test = self.data_builder('/data/',
                                          train=False, download=True,
                                          transform=torchvision.transforms.Compose(
                                              self.transforms)
                                          )

        self.classes = self.train.classes
        self.n_tasks = int(len(self.train.classes)/2)

    def _get_classes(self, task: int):
        return task*2, task*2+1

    def train_data(self, task: int = None,loader=True, batch_size=None) -> DataLoader:
        batch_size=self.batch_size if batch_size==None else batch_size
        if(task == None):
            return DataLoader(self.train, batch_size=batch_size, shuffle=True)

        cl_a, cl_b = self._get_classes(task)

        mask = (self.train.targets ==
                cl_a) | (self.train.targets == cl_b)


        if(loader):
            indexes = torch.arange(0, len(mask))[mask]

            subset = torch.utils.data.Subset(self.train, indexes)

            return DataLoader(subset, batch_size=batch_size, shuffle=True)
        else:
            return self.train.data[mask]

    def test_data(self, task: int = None) -> DataLoader:
        if(task == None):
            return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)
        cl_a, cl_b = self._get_classes(task)

        mask = (self.test.targets == cl_a) | (
            self.test.targets == cl_b)

        indexes = torch.arange(0, len(mask))[mask]

        subset = torch.utils.data.Subset(self.test, indexes)

        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)
