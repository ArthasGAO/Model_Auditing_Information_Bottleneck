from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import MNIST
import sys
import os


class MNISTDataset(Dataset):
    # Singleton
    mnist_train = None
    mnist_val = None

    def __init__(
            self
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None

        # Singletons of MNIST train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        if MNISTDataset.mnist_train is None:
            MNISTDataset.mnist_train = MNIST(
                f"{sys.path[0]}/data", train=True, download=True, transform=Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, ), (0.5, ))
                                                                    ])
            )

        if MNISTDataset.mnist_val is None:
            MNISTDataset.mnist_val = MNIST(
                f"{sys.path[0]}/data", train=False, download=True, transform=Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, ), (0.5, ))
                                                                    ])
            )

        self.train_set = MNISTDataset.mnist_train
        self.test_set = MNISTDataset.mnist_val