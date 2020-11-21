import os
from pathlib import Path
import h5py
import logging
from typing import Union, List, Callable

import numpy as np
import tensorflow as tf
import torch as th
from torch.utils.data import Dataset


class TorchEMNISTFed(Dataset):
    data_dir = Path("data/emnist_fed")

    def __init__(self, split: Union[str, List[str]], transform: Callable = None):
        self._download_all_data()
        self.images, self.labels = self._get_data(split)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.images[idx, ])
        else:
            img = th.tensor(self.images[idx, ])

        return img, th.tensor(self.labels[idx, ]).long()

    @classmethod
    def get_client_ids(cls, data_set):
        cls._download_all_data()
        ext = "_img.npy"
        return [
            f[: -len(ext)]
            for f in os.listdir(cls.data_dir / data_set)
            if f.endswith(ext)
        ]
    
    @classmethod
    def _download_all_data(cls):
        is_extract = not cls.data_dir.exists()
        if not is_extract:
            return

        logging.info(f"Data extration: {is_extract} ...")
        filename = "fed_emnist.tar.bz2"
        path = tf.keras.utils.get_file(
            filename,
            origin="https://storage.googleapis.com/tff-datasets-public/" + filename,
            file_hash="fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0",
            hash_algorithm="sha256",
            extract=True,
            archive_format="tar",
        )
        logging.info(f"Data extrated")

        dir_path = os.path.dirname(path)

        def download_split(data_set):
            split_dir = cls.data_dir / data_set
            try:
                os.makedirs(split_dir)
            except FileExistsError:
                return
            h5 = h5py.File(
                os.path.join(dir_path, f"fed_emnist_{data_set}.h5"), "r"
            )["examples"]

            for client_id in h5.keys():
                images = np.expand_dims(h5[client_id]["pixels"], axis=1)
                labels = np.array(h5[client_id]["label"])
                np.save(split_dir / f"{client_id}_img", images)
                np.save(split_dir / f"{client_id}_label", labels)

        download_split("train")
        download_split("test")

    def _get_client_data(self, data_set, client_id):
        split_dir = self.data_dir / data_set

        images = np.load(split_dir / f"{client_id}_img.npy")
        labels = np.load(split_dir / f"{client_id}_label.npy")
        return images, labels

    def _get_data(self, split):
        if isinstance(split, str):
            data_set = split
            client_ids = self.get_client_ids(data_set)
        else:
            data_set = "train"
            client_ids = split

        images = []
        labels = []
        for c in client_ids:
            i, l = self._get_client_data(data_set, c)
            images.append(i)
            labels.append(l)

        return np.concatenate(images), np.concatenate(labels)
