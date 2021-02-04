from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.data_transforms import data_transforms


class Mnist_DataLoader(BaseDataLoader):
    """
    MNIST data loading using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=transforms
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class Cifar10_DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):

        # Define data transformations
        transforms = data_transforms()

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(
            self.data_dir, train=training, download=True, transform=transforms
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
