import torch
from torch import nn, optim
import torch.nn.functional as F

def get_mask(input, target, device, n_nabels):
    tasks = (target/2).type(torch.IntTensor).unsqueeze(-1).expand(-1, n_nabels).to(device)
    tasks_base = (torch.arange(0, n_nabels) /
                  2).expand(input.shape[0], -1).type(torch.IntTensor).to(device)

    mask = torch.empty((input.shape[0], n_nabels), dtype=torch.float32).to(device)

    mask[tasks != tasks_base] = -float("Inf")
    mask[tasks == tasks_base] = 0

    return mask


def train(epochs: int, model, n_nabels, loader: torch.utils.data.DataLoader, optimizer, device=torch.device("cpu"), print_delay=64, loss_fn=F.nll_loss, printer=True,regularizer_fn=None,lambda_reg=0):
    loader.pin_memory = True
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(loader):
            inputs, target = data[0].to(device), data[1].to(device)
            mask = get_mask(inputs, target, device, n_nabels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            outputs = F.log_softmax(outputs+mask)

            loss = loss_fn(outputs, target)
            if(regularizer_fn!=None):
                loss+=lambda_reg*regularizer_fn()

            loss.backward()
            optimizer.step()

            # print statistics
            if(printer):
                running_loss += loss.item()
                if print_delay != None and i % print_delay == print_delay-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


def test(model, n_nabels, loader: torch.utils.data.DataLoader, device=torch.device("cpu"), loss_fn=F.nll_loss, printer=True):
    model.eval()
    test_loss = 0
    correct = 0
    loader.pin_memory = True
    with torch.no_grad():
        for d, t in loader:
            data = d.to(device)
            target = t.to(device)

            mask = get_mask(data, target, device, n_nabels)

            output = model(data)

            output = F.log_softmax(output+mask)

            test_loss += loss_fn(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    if(printer):
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
    return test_loss, accuracy
