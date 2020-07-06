from comet_ml import Experiment

from typing import Tuple, List, Callable
import logging

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from FLF.TorchFederatedLearner import TorchFederatedLearner, TorchFederatedLearnerConfig


class TorchFederatedLearnerMNISTConfig(TorchFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.


class TorchFederatedLearnerMNIST(TorchFederatedLearner):
    def __init__(
        self, experiment: Experiment, config: TorchFederatedLearnerMNISTConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {TorchFederatedLearnerMNISTConfig} -- Training configuration description.
        """
        super().__init__(experiment, config)
        self.config = config  # Purly to help intellisense

    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader]:
        logging.info("MNIST data loading ...")
        minist_train_ds, mnist_test_ds = self.__get_mnist()
        logging.info("MNIST data loaded.")

        logging.info("Data for client is being sampled ...")
        n_training_samples = len(minist_train_ds)
        logging.info("Number of training samples: {n_training_samples}")
        if self.config.IS_IID_DATA:
            indices = np.arange(n_training_samples)
            indices = indices.reshape(self.config.N_CLIENTS, -1)
            indices = indices.tolist()
        else:
            indices = self.__distribute_data_non_IID(minist_train_ds)

        train_loader_list = []
        for idx in indices:
            sampler = th.utils.data.sampler.SubsetRandomSampler(idx)
            # sampler = th.utils.data.sampler.SequentialSampler(idx)
            loader = th.utils.data.DataLoader(
                dataset=minist_train_ds,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.DL_N_WORKER,
                # pin_memory=True,
                sampler=sampler,
            )
            train_loader_list.append(loader)
        logging.info("Data for client is sampled.")

        test_loader = th.utils.data.DataLoader(
            mnist_test_ds, batch_size=64, num_workers=self.config.DL_N_WORKER,
        )

        return train_loader_list, test_loader, 0.1

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
            digit_sort_idx[i : i + 2, ].flatten()
            for i in range(0, 2 * self.config.N_CLIENTS, 2)
        ]
        return indices

    def get_model_cls(self) -> Callable[[], nn.Module]:
        return Net, True

    def get_loss(self) -> nn.Module:
        return F.nll_loss


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
