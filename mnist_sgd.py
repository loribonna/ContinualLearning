import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from settings import Mode
from torchvision.datasets import MNIST

mode = Mode.domain_il

batch_size, d_in, d_hidden, d_out = 128, 28*28, 100, 10
lr, momentum = 0.03, 0
n_tasks_domain_il=20
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
        x = self.fc3(x)
        return F.log_softmax(x)


model = Net(device)
model.to(device)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr, momentum)

if(mode == Mode.task_il):
    from task_il_utils import train, test
    from datasets.sequentializer import Sequentializer

    dataset = Sequentializer(MNIST, batch_size=batch_size, transforms=[
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])

    for task in range(dataset.n_tasks):
        print("-- TASK %d" % task)
        train(epochs, model, 10, dataset.train_data(
            task), optimizer, device)
        test(model, 10, dataset.test_data(task), device)
    test_loss, accuracy = test(model, 10, dataset.test_data(), device)

elif(mode == Mode.class_il):
    from mlp_utils import train, test
    from datasets.sequentializer import Sequentializer

    dataset = Sequentializer(MNIST, batch_size=batch_size, transforms=[
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])

    for task in range(dataset.n_tasks):
        print("-- TASK %d" % task)
        train(epochs, model, dataset.train_data(
            task), optimizer, device)
        test(model, dataset.test_data(task), device)
    test(model, dataset.test_data(), device)
else:
    from mlp_utils import train, test
    from datasets.permuter import Permuter
    
    dataset = Permuter(MNIST,n_tasks_domain_il, batch_size)

    test(model, dataset.test_data(), device)
    for task in range(dataset.n_tasks):
        print("-- TASK %d" % task)
        train(epochs, model, dataset.train_data(
            task), optimizer, device)
        test(model, dataset.test_data(task), device)
    test(model, dataset.test_data(), device)
