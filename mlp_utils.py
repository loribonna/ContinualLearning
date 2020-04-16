import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import datasets, transforms


def train(epochs: int, model, loader: torch.utils.data.DataLoader, optimizer, device=torch.device("cpu"), print_delay=64, loss_fn=F.nll_loss, task=None, regularizer_fn=None, lambda_reg=0):
    loader.pin_memory = True
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(loader):
            inputs, target = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, target)
            if(regularizer_fn != None):
                loss += lambda_reg*regularizer_fn()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if print_delay != None and i % print_delay == print_delay-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(model, loader: torch.utils.data.DataLoader, device=torch.device("cpu"), loss_fn=F.nll_loss, task=None):
    model.eval()
    test_loss = 0
    correct = 0
    loader.pin_memory = True
    with torch.no_grad():
        for d, t in loader:
            data = d.to(device)
            target = t.to(device)

            output = model(data)
            test_loss += loss_fn(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    return test_loss, accuracy
