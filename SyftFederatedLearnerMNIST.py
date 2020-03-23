from comet_ml import Experiment

from typing import Tuple
import logging

import numpy as np
import syft as sy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from syftutils.datasets import get_dataset_items_at

from SyftFederatedLearner import SyftFederatedLearner, SyftFederatedLearnerConfig


class SyftFederatedLearnerMNISTConfig(SyftFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.


class SyftFederatedLearnerMNIST(SyftFederatedLearner):
    def __init__(
        self, experiment: Experiment, config: SyftFederatedLearnerMNISTConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {SyftFederatedLearnerMNISTConfig} -- Training configuration description.
        """
        super().__init__(experiment, config)
        self.config = config  # Purly to help intellisense

    def load_data(self) -> Tuple[sy.FederatedDataLoader, th.utils.data.DataLoader]:
        logging.info("MNIST data loading ...")
        minist_train_ds, mnist_test_ds = self.__get_mnist()
        logging.info("MNIST data loaded.")

        logging.info("Data distributing ...")
        if self.config.IS_IID_DATA:
            federated_train_dataset = minist_train_ds.federate(
                self.clients
            )  # TODO HARD get list of index samples instead
        else:
            federated_train_dataset = self.__distribute_data_non_IID(
                minist_train_ds
            )  # TODO get list of index samples instead

        # TODO HARD use list of DataLoader and indices with sampler
        federated_train_loader = sy.FederatedDataLoader(
            federated_train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.DL_N_WORKER,
            pin_memory=True,
        )
        logging.info("Data distributed.")

        test_loader = th.utils.data.DataLoader(
            mnist_test_ds,
            batch_size=64,
            shuffle=True,
            num_workers=self.config.DL_N_WORKER,
            pin_memory=True,
        )

        return federated_train_loader, test_loader

    def __get_mnist(self):
        minist_train_ds = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        mnist_test_ds = datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return minist_train_ds, mnist_test_ds

    def __distribute_data_non_IID(self, minist_train_ds):
        digit_sort_idx = np.concatenate(
            [np.where(minist_train_ds.targets == i)[0] for i in range(10)]
        )
        digit_sort_idx = digit_sort_idx.reshape(2 * self.config.N_CLIENTS, -1)
        np.random.shuffle(digit_sort_idx)
        indices = [
            digit_sort_idx[i : i + 2,].flatten()
            for i in range(0, 2 * self.config.N_CLIENTS, 2)
        ]
        # TODO return idices and remove the rest
        dss = []
        for idx, c in zip(indices, self.clients):
            data, target = get_dataset_items_at(minist_train_ds, idx)
            dss.append(sy.BaseDataset(data.send(c), target.send(c)))

        federated_train_dataset = sy.FederatedDataset(dss)
        return federated_train_dataset

    def build_model(self) -> nn.Module:
        return Net()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
