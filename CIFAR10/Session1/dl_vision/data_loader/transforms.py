import abc
import torchvision.transforms as transforms


class TransformsFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_trainset() if train else self.build_testset()

        @abc.abstractmethod
        def build_trainset(self):
            pass

        @abc.abstractmethod
        def build_testset(self):
            pass


class CIFAR10Transforms(TransformsFactoryBase):
    def build_trainset(self):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.20110)
                ),
            ]
        )

    def build_testset(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
