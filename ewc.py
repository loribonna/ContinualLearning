import torch
from torch import nn, optim
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms
from datasets.sequentializer import Sequentializer
import numpy as np
from task_il_utils import get_mask
from settings import Mode

mode = Mode.domain_il

batch_size, d_in, d_hidden, d_out = 128, 28*28, 100, 10
lr, momentum = 0.1, 0
lambda_reg = 50000
epochs = 1
n_tasks_domain_il=20
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

        self.fisher = np.array([])
        self.prev_parameters = np.array([])

    def forward(self, x):
        x = x.view(-1, d_in)  # flatten out

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def _init_fisher(self):
        tmp_fisher = {}
        for name, param in self.named_parameters():
            tmp_fisher[name] = torch.zeros_like(param.detach(), device=device)
        return tmp_fisher

    def _init_prev_params(self):
        prev_parameters = {}
        for name, param in self.named_parameters():
            prev_parameters[name] = torch.tensor(
                param.detach(), device=self.device)
        return prev_parameters

    def get_regularizer(self):
        tot = 0
        for i in range(len(self.fisher)):
            for name, param in self.named_parameters():
                tot += torch.sum(torch.sum(self.fisher[i][name] *
                                           ((param-self.prev_parameters[i][name])**2)))
        return tot/2.

    def estimate_fisher(self, train_loader: torch.utils.data.DataLoader):
        self.eval()
        n_train=0
        tmp_fisher = self._init_fisher()

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            n_train+=inputs.shape[0]
            self.zero_grad()

            outputs = self.forward(inputs)

            loss = F.nll_loss(F.log_softmax(outputs, dim=1),
                              torch.max(outputs, axis=1)[1])
            loss.backward()

            for name, param in self.named_parameters():
                tmp_fisher[name] += param.grad.detach()**2

        tmp_params = {}
        for n, p in self.named_parameters():
            tmp_params[n] = torch.tensor(p.detach(), device=self.device)
            tmp_fisher[n] = tmp_fisher[n]/n_train

        self.fisher = np.append(self.fisher, tmp_fisher)
        self.prev_parameters = np.append(self.prev_parameters, tmp_params)


model = Net(device, F.nll_loss)
model.to(device)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr, momentum)

def train(epochs: int, model, n_nabels, loader: torch.utils.data.DataLoader, optimizer, print_delay=5000, printer=True):
    loader.pin_memory = True
    for epoch in range(epochs):
        running_loss = 0.0
        print_tot = 1
        model.train()
        for i, data in enumerate(loader):
            inputs, target = data[0].to(device), data[1].to(device)

            if mode == Mode.task_il:
                mask = get_mask(inputs, target, device, n_nabels)

            optimizer.zero_grad()

            outputs = model(inputs)

            if mode == Mode.task_il:
                outputs = outputs+mask

            outputs = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(outputs, target) + \
                lambda_reg * model.get_regularizer()

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

            if mode == Mode.task_il:
                mask = get_mask(data, target, device, n_nabels)

            output = model(data)

            if mode == Mode.task_il:
                output = output+mask

            output = F.log_softmax(output, dim=1)

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


if (mode==Mode.task_il or mode==Mode.class_il):
    dataset = Sequentializer(MNIST, batch_size=batch_size, transforms=[
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])
else:
    from datasets.permuter import Permuter
    
    dataset = Permuter(MNIST,n_tasks_domain_il, batch_size)


for task in range(dataset.n_tasks):
    print("-- TASK %d" % task)
    train(epochs, model, 10, dataset.train_data(task), optimizer)
    test(model, 10, dataset.test_data(task))
    print("\t-- Estimate fisher")
    model.estimate_fisher(dataset.train_data(task))

test_loss, accuracy = test(model, 10, dataset.test_data())
