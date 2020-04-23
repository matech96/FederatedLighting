import os
import h5py
from typing import Union, List, Callable

import tensorflow as tf
from torch.utils.data import Dataset


class TorchCIFAR10Fed(Dataset):
    N_ELEMENTS_PER_CLIENT = 100

    def __init__(self, split: Union[str, List[str]], transform: Callable = None):
        if isinstance(split, str) and split == "test":
            self.h5 = get_cifar100fed_h5("test")
            self.client_ids = list(self.h5.keys())
        else:
            self.h5 = get_cifar100fed_h5("train")
            self.client_ids = split

        self.images = []
        self.labels = []
        for client_id in self.client_ids:
            for item_id in range(100):
                self.images.append(self.h5[client_id]["image"][item_id])
                self.labels.append(self.h5[client_id]["label"][item_id])
        # self.cliens = [self.h5[id] for id in self.client_ids]
        self.transform = transform

    def __len__(self):
        return len(self.client_ids) * self.N_ELEMENTS_PER_CLIENT

    def __getitem__(self, idx):
        # client_id = idx // self.N_ELEMENTS_PER_CLIENT
        # item_id = idx % self.N_ELEMENTS_PER_CLIENT

        # client = self.cliens[client_id]
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_cifar100fed_h5(split):
    dir_path = os.path.dirname(get_cifar100fed_h5.path)
    return h5py.File(os.path.join(dir_path, f"fed_cifar100_{split}.h5"), "r")[
        "examples"
    ]


get_cifar100fed_h5.path = tf.keras.utils.get_file(
    "fed_cifar100.tar.bz2",
    origin="https://storage.googleapis.com/tff-datasets-public/fed_cifar100.tar.bz2",
    file_hash="e8575e22c038ecef1ce6c7d492d7abee7da13b1e1ba9b70a7fc18531ba7590de",
    hash_algorithm="sha256",
    extract=False,#True,
    archive_format="tar",
)
