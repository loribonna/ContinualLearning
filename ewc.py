import torch
from torch import nn, optim
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms
from datasets.mnist_sequential import SequentialMNIST
import numpy as np

batch_size, d_in, d_hidden, d_out = 64, 28*28, 100, 10
lr, momentum = 0.1, 0.1
lambda_reg = 500000
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True


class Net(nn.Module):
    def __init__(self, device, loss_fn):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.device = device
        self.loss_fn = loss_fn

        self.to(device)

        self._get_c_params()
        self.fisher = np.array([])

        self.tmp_fisher = self._init_fisher()
        self.fisher = np.append(self.fisher, self._init_fisher())

    def forward(self, x):
        x = x.view(-1, d_in)  # flatten out

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

    def _init_fisher(self):
        tmp_fisher = {}
        for name, param in self.named_parameters():
            tmp_fisher[name] = torch.zeros_like(param, device=device)
        return tmp_fisher

    def _get_c_params(self):
        self.prev_parameters = {}
        for name, param in self.named_parameters():
            self.prev_parameters[name] = torch.tensor(param, device=self.device)

    def get_regularizer(self):
        tot = 0
        for name, param in self.named_parameters():
            for fisher in self.fisher:
                tot += torch.sum(torch.sum(fisher[name] *
                                           ((param-self.prev_parameters[name])**2)))
        return tot/2.

    def _swap_params(self, new_params):
        for name, param in self.named_parameters():
            param = new_params[name]

    def estimate_fisher(self, train_loader: torch.utils.data.DataLoader):
        # train_loader.batch_size=1
        self.eval()
        n_train = len(train_loader)*(train_loader.batch_size-1)
        tmp_fisher = self._init_fisher()

        for inputs, targets in train_loader:
            inputs=inputs.to(self.device)

            self.zero_grad()

            outputs = self.forward(inputs)

            loss = F.nll_loss(F.log_softmax(outputs,dim=-1), torch.max(outputs, axis=1)[1])
            loss.backward()

            for name, param in self.named_parameters():
                tmp_fisher[name] += torch.sum(param.grad.detach()**2)

        for n, p in self.named_parameters():
            self.prev_parameters[n]=p.clone().to(self.device)
            tmp_fisher[n] = tmp_fisher[n]/n_train

        self.fisher = np.append(self.fisher, tmp_fisher)


dataset = SequentialMNIST(MNIST, batch_size=batch_size, transforms=[
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])


model = Net(device, F.nll_loss)
model.to(device)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr, momentum)

task_il = True


def get_mask(input, target, device, n_nabels):
    tasks = (target/2).type(torch.IntTensor).unsqueeze(-1).expand(-1,
                                                                  n_nabels).to(device)
    tasks_base = (torch.arange(0, n_nabels) /
                  2).expand(input.shape[0], -1).type(torch.IntTensor).to(device)

    mask = torch.empty((input.shape[0], n_nabels),
                       dtype=torch.float32).to(device)

    mask[tasks != tasks_base] = -float("Inf")
    mask[tasks == tasks_base] = 0

    return mask


def train(epochs: int, model, n_nabels, loader: torch.utils.data.DataLoader, optimizer, print_delay=5000, printer=True):
    loader.pin_memory = True
    for epoch in range(epochs):
        running_loss = 0.0
        print_tot = 1
        model.train()
        for i, data in enumerate(loader):
            inputs, target = data[0].to(device), data[1].to(device)

            if task_il:
                mask = get_mask(inputs, target, device, n_nabels)

            optimizer.zero_grad()

            outputs = model(inputs)

            if task_il:
                outputs = outputs+mask

            outputs = F.log_softmax(outputs,dim=-1)

            loss = F.nll_loss(outputs, target) + lambda_reg*model.get_regularizer()

            loss.backward()
            optimizer.step()

            # print statistics
            if(printer):
                running_loss += loss.item()
                if print_delay != None and i*loader.batch_size >= print_delay*print_tot:    # print every 2000 mini-batches
                    print_tot += 1
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i*loader.batch_size, running_loss / 2000))
                    running_loss = 0.0


def test(model, n_nabels, loader: torch.utils.data.DataLoader, printer=True):
    model.eval()
    test_loss = 0
    correct = 0
    loader.pin_memory = True
    with torch.no_grad():
        for d, t in loader:
            data = d.to(device)
            target = t.to(device)

            if task_il:
                mask = get_mask(data, target, device, n_nabels)

            output = model(data)

            if task_il:
                output = output+mask

            output = F.log_softmax(output,dim=-1)

            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    if(printer):
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
    return test_loss, accuracy


for task in range(dataset.n_tasks):
    print("-- TASK %d" % task)

    train_loader = dataset.train_data(task)

    train(epochs, model, 10, train_loader, optimizer)
    test(model, 10, dataset.test_data(task))
    model.estimate_fisher(train_loader)

test_loss, accuracy = test(model, 10, dataset.test_data())
