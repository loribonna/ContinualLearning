import torch
from torch import nn, optim
from sequentializer import TaskILDataset, ClassILDataset
from torchvision.datasets import MNIST
from mlp import train, test
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

batch_size, d_in, d_hidden, d_out = 64, 28*28+1, 100, 2
lr, momentum = 1e-4, 0.1
epochs = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True


class InsertTaskID(nn.Module):
    def forward(self, x, task_id):
        return torch.stack((x, task_id))


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28+1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.device = device

    def forward(self, x):
        x = x.view(-1, 28*28+1)  # flatten out

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


dataset = TaskILDataset(MNIST, batch_size=batch_size, transforms=[
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

# model = Net(device)
# model.to(device)

# optimizer = optim.SGD(model.parameters(), lr, momentum)

# print("NON-SEQUENTIAL")

# test(model, dataset.test_data(), device)
# train(epochs, model, dataset.train_data(), optimizer, device)
# test(model, dataset.test_data(), device)

print("SEQUENTIAL")
# del model


model = Net(device)
model.to(device)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr, momentum)

for task in range(dataset.n_tasks):
    print("-- TASK %d" % task)
    train(epochs, model, dataset.train_data(
        task), optimizer, device)
    test(model, dataset.test_data(task), device)
test(model, dataset.test_data(), device)
