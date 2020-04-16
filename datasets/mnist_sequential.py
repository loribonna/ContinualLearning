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


class SequentialMNIST:  # each task with 2 classes
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

    def train_data(self, task: int = None,loader=True) -> DataLoader:
        if(task == None):
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

        cl_a, cl_b = self._get_classes(task)

        mask = (self.train.targets ==
                cl_a) | (self.train.targets == cl_b)


        if(loader):
            indexes = torch.arange(0, len(mask))[mask]

            subset = torch.utils.data.Subset(self.train, indexes)

            return DataLoader(subset, batch_size=self.batch_size, shuffle=True)
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



class TaskILMNIST:
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

        self.n_tasks = int(len(self.train.classes)/2)
        self.task_train_mask = self.train.targets/2
        self.task_test_mask = self.test.targets/2

        self.train.data, self.train.targets = self._add_task_id(
            self.train.data, self.train.targets)
        self.test.data, self.test.targets = self._add_task_id(
            self.test.data, self.test.targets)

        self.train.data = self.train.data.unsqueeze(-1)  # weird bug
        self.test.data = self.test.data.unsqueeze(-1)  # weird bug

    def _add_task_id(self, data, targets):
        data = data.view(data.shape[0], -1)

        np_targets = targets.numpy()
        np_tasks = (np.copy(np_targets)/2).astype(data.numpy().dtype)

        mask = (np_tasks*2 == np_targets)
        np_targets[mask] = 0
        np_targets[~mask] = 1

        tasks = torch.from_numpy(np_tasks)
        return torch.cat((data, tasks.unsqueeze(-1)), 1), torch.from_numpy(np_targets)

    def train_data(self, task: int = None) -> DataLoader:
        if(task == None):
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

        indexes = torch.arange(0, len(self.task_train_mask))[self.task_train_mask==task]

        subset = torch.utils.data.Subset(self.train, indexes)

        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def test_data(self, task: int = None) -> DataLoader:
        if(task == None):
            return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)

        indexes = torch.arange(0, len(self.task_test_mask))[self.task_test_mask==task]

        subset = torch.utils.data.Subset(self.test, indexes)

        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

