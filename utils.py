import torchvision
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def print_loader(loader):
    it = iter(loader)
    d, t=it.next()

    imshow(torchvision.utils.make_grid(d))