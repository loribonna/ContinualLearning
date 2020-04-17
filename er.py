import torch
from torch import nn, optim
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms
from datasets.mnist_sequential import SequentialMNIST
from task_il_utils import get_mask
import numpy as np

batch_size, d_in, d_hidden, d_out = 32, 28*28, 100, 10
lr, momentum = 0.1, 0
lambda_reg = 2.8 
buff_size = 5120
n_buf_samples = 32
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

task_il = False


class Net(nn.Module):
    def __init__(self, device, buff_size=10, reservoir=True):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.device = device
        self.reservoir = reservoir

        self.buffer_size = buff_size
        self.n_seen_elements = 0

    def forward(self, x):
        x = x.view(-1, d_in)  # flatten out

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

    def add_batch_buffer(self, inputs, targets):
        pusher = self._push_reservoir if self.reservoir else self._push_herding
        for index in range(len(inputs)):
            pusher(inputs[index], targets[index])

    def _push_reservoir(self, data, target):
        self.n_seen_elements += 1

        if(self.n_seen_elements == 1):  # initialize
            self.buffer_data = torch.tensor(
                data.clone(), device=self.device).unsqueeze(0)  # 2 dim
            self.buffer_targets = torch.tensor(
                target.clone(), device=self.device).unsqueeze(0)
            return

        if(self.n_seen_elements < self.buffer_size+1):  # have space
            self.buffer_data = torch.cat(
                (self.buffer_data, data.clone().unsqueeze(0)))
            self.buffer_targets = torch.cat(
                (self.buffer_targets, target.clone().unsqueeze(0)))
        else:  # reservoir
            r = torch.randint(self.n_seen_elements-1, (1,))
            if(r < self.buffer_size):
                self.buffer_data[r] = data.clone()
                self.buffer_targets[r] = target.clone()

    def sample_buffer(self, n_elements=None):
        if(self.n_seen_elements == 0):
            return None, None

        max_elements = np.min((self.buffer_size, self.n_seen_elements))
        # max_elements = torch.min(torch.tensor((self.buffer_size, self.n_seen_elements)))
        n_elements = max_elements if n_elements == None else np.min((n_elements, max_elements))

        indexes = np.random.choice(max_elements, size=n_elements, replace=False)

        return self.buffer_data[indexes], self.buffer_targets[indexes]

    def _push_herding(self):
        raise "Heriding Not Implemented"


dataset = SequentialMNIST(MNIST, batch_size=batch_size, transforms=[
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])


model = Net(device, buff_size)
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
            inputs, targets = data[0].to(device), data[1].to(device)

            buf_inputs, buf_targets = model.sample_buffer(n_buf_samples)

            # extended_inputs = torch.cat((inputs, buf_inputs))
            # extended_targets = torch.cat((targets, buf_targets))

            if task_il:
                mask = get_mask(inputs, targets, device, n_nabels)
                mask_buf = get_mask(
                    buf_inputs, buf_targets, device, n_nabels) if buf_inputs != None else None

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs_buff = model(buf_inputs) if buf_inputs != None else None

            if task_il:
                outputs = outputs+mask
                outputs_buff = outputs_buff+mask_buf if buf_inputs != None else None

            outputs = F.log_softmax(outputs, dim=1)
            outputs_buff = F.log_softmax(outputs_buff, dim=1) if buf_inputs != None else None

            loss = F.nll_loss(outputs, targets)
            loss_buf = F.nll_loss(outputs_buff, buf_targets) if buf_inputs != None else 0

            loss = loss + lambda_reg*loss_buf

            loss.backward()
            optimizer.step()

            model.add_batch_buffer(inputs, targets)

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


for task in range(dataset.n_tasks):
    print("-- TASK %d" % task)
    train(epochs, model, 10, dataset.train_data(
        task), optimizer)
    test(model, 10, dataset.test_data(task))
test(model,10, dataset.test_data())
