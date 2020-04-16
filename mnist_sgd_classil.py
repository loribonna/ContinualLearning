import torch
from torch import nn, optim
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms
from datasets.mnist_sequential import ClassILDataset
from mlp_utils import train, test

batch_size, d_in, d_hidden, d_out = 64, 28*28, 100, 10
lr, momentum = 1e-4, 0.1
epochs = 4
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
        x = self.fc3(x)
        return F.log_softmax(x)


dataset = SequentialMNIST(MNIST, batch_size=batch_size, transforms=[
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])


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
