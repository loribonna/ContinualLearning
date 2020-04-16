import torch
from torch import nn, optim
from datasets.mnist_sequential import SequentialMNIST
from torchvision.datasets import MNIST
import torch.nn.functional as F
from task_il_utils import train, test
import numpy as np
from torchvision import transforms
import os
import multiprocessing

batch_size, d_in, d_hidden, d_out = 10, 28*28, 100, 10
lr, momentum = 0.1, 0
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.device = device

    def forward(self, x):
        x = x.view(-1, d_in)  # flatten out

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        # return F.log_softmax(self.fc3(x))

def run():
    print("RUNNING {}".format(os.getpid()))
    dataset = SequentialMNIST(MNIST, batch_size=batch_size, transforms=[
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])


    model = Net(device)
    model.to(device)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr, momentum)

    for task in range(dataset.n_tasks):
        # print("-- TASK %d" % task)
        train(epochs, model, 10, dataset.train_data(
            task), optimizer, device)
        test(model,10, dataset.test_data(task), device)
    test_loss, accuracy = test(model,10, dataset.test_data(), device)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy:{:.0f}%\n'.format(
        test_loss, accuracy))

for i in range(5):
    run()