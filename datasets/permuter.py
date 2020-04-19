import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Iterator
from torch.utils.data import DataLoader

initial_seed = 1


class PermuteTransform:
    def __init__(self, perm):
        self.perm = perm

    def __call__(self, img):
        img_shape = img.shape
        img = img.view(-1)

        return img[self.perm].view(img_shape)


class ApplyPermutationsTransform:
    def __init__(self, permutations):
        self.n_tasks = len(permutations)
        self.permutations=permutations

    def __call__(self, img):
        task = torch.randint(self.n_tasks, size=(1,))
        return PermuteTransform(self.permutations[task.item()])(img)


class Permuter:
    def __init__(self, data_builder, n_tasks, batch_size, transforms=[torchvision.transforms.ToTensor()], no_cuda=False):
        self.data_builder = data_builder
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_tasks = n_tasks
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self._setup()

    def _setup(self):
        if (self.data_builder):
            self.train = self.data_builder('/data/',
                                           train=True, download=True,
                                           transform=torchvision.transforms.Compose(self.transforms)
                                           )
            self.test = self.data_builder('/data/',
                                          train=False, download=True,
                                           transform=torchvision.transforms.Compose(self.transforms)
                                          )

        self.classes = self.train.classes
        self.permutations = []
        for task in range(self.n_tasks):
            generator = torch.Generator(device=self.train.data.device)
            generator.manual_seed(task*initial_seed)
            indexes = torch.randperm(
                self.train.data.shape[1]*self.train.data.shape[2], generator=generator, device=self.train.data.device)

            self.permutations.append(indexes)

    def train_data(self, task: int) -> DataLoader:
        transforms = [*self.transforms, PermuteTransform(perm=self.permutations[task])]
        self.train.transform = torchvision.transforms.Compose(transforms)

        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def test_data(self, task: int = None) -> DataLoader:
        if(task != None):
            transforms = [*self.transforms, PermuteTransform(perm=self.permutations[task])]
            self.test.transform = torchvision.transforms.Compose(transforms)

            return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)
        else:
            transforms = [*self.transforms, ApplyPermutationsTransform(permutations=self.permutations)]
            self.test.transform = torchvision.transforms.Compose(transforms)

            return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)
