import os

import torchvision
import torchvision.transforms as transforms


class downloadFiles:
    def __init__(self):
        self.download_mnist()

    def download_mnist(self):
        directory = os.path.abspath(os.path.join(os.getcwd(), "Datasets"))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torchvision.datasets.FashionMNIST(root=directory, train=True, download=True,
                                          transform=transforms.ToTensor())
        torchvision.datasets.FashionMNIST(root=directory, train=False, download=True,
                                          transform=transforms.ToTensor())


if __name__ == '__main__':
    downloadFiles = downloadFiles()
