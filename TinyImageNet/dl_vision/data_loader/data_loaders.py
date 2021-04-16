from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile
from tqdm.auto import tqdm, trange
import glob

from utils.util import download_and_extract_archive

# from base.base_data_loader import BaseDataLoader
# class MnistDataLoader1(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MNISTDataLoader:
    def __init__(
        self,
        transforms,
        data_dir,
        batch_size=64,
        shuffle=True,
        nworkers=2,
        pin_memory=True,
        validation_split=0.0,
    ):
        self.validation_split = validation_split
        self.data_dir = data_dir

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.train_set = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True),
        )

        self.test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False),
        )

        self.init_kwargs = {
            "shuffle": shuffle,
            "batch_size": batch_size,
            "num_workers": nworkers,
            "pin_memory": pin_memory,
        }

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(
            self.test_set, **self.init_kwargs
        )


class CIFAR10DataLoader:

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ):

        self.data_dir = data_dir

        self.train_dataset = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True),
        )

        self.test_dataset = datasets.CIFAR10(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False),
        )

        self.init_kwargs = {
            "shuffle": shuffle,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    def get_loaders(self):
        return (
            DataLoader(self.train_dataset, **self.init_kwargs),
            DataLoader(self.test_dataset, **self.init_kwargs),
        )


class TinyImageNetDataLoader:
    def __init__(
        self,
        transforms,
        data_dir,
        batch_size=64,
        shuffle=True,
        nworkers=2,
        pin_memory=True,
        validation_split=0.0
    ):
        self.data_dir = data_dir

        self.train_set = TinyImageNet(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True),
        )

        self.test_set = TinyImageNet(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False),
        )

        self.init_kwargs = {
            "shuffle": shuffle,
            "batch_size": batch_size,
            "num_workers": nworkers,
            "pin_memory": pin_memory,
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(
            self.test_set, **self.init_kwargs
        )


class TinyImageNet(Dataset):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    dataset_folder_name = "tiny-imagenet-200"

    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = "val_annotations.txt"

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download and (
            not os.path.isdir(os.path.join(self.root, self.dataset_folder_name))
        ):
            self.download()

        self.split_dir = "train" if train else "val"
        self.split_dir = os.path.join(
            self.root, self.dataset_folder_name, self.split_dir
        )
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self.split_dir, "**", "*.%s" % self.EXTENSION),
                recursive=True,
            )
        )

        self.target = []
        self.labels = {}

        # build class label - number mapping
        with open(
            os.path.join(self.root, self.dataset_folder_name, self.CLASS_LIST_FILE), "r"
        ) as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        # build labels for NUM_IMAGES_PER_CLASS images
        if train:
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels[f"{label_text}_{cnt}.{self.EXTENSION}"] = i

        # build the validation dataset
        else:
            with open(
                os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), "r"
            ) as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        self.target = [
            self.labels[os.path.basename(filename)] for filename in self.image_paths
        ]

    def download(self):
        download_and_extract_archive(self.url, self.root, filename=self.filename)

    def __getitem__(self, index):
        filepath = self.image_paths[index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_paths)
