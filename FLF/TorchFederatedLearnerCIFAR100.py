from comet_ml import Experiment

import logging
from typing import Tuple, List, Callable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from FLF.TorchFederatedLearner import TorchFederatedLearner, TorchFederatedLearnerConfig
from FLF.data.TorchCIFAR100Fed import TorchCIFAR100Fed


class TorchFederatedLearnerCIFAR100Config(TorchFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.


class TorchFederatedLearnerCIFAR100(TorchFederatedLearner):
    N_TRAINING_CLIENTS = 500

    def __init__(
        self, experiment: Experiment, config: TorchFederatedLearnerCIFAR100Config
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {TorchFederatedLearnerCIFAR100Config} -- Training configuration description.
        """
        super().__init__(experiment, config)
        self.config = config  # Purly to help intellisense

    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if self.config.IS_IID_DATA:
            train_loader_list = self.get_iid_data(transform)
        else:
            train_loader_list = self.get_non_iid_data(transform)

        test_loader = th.utils.data.DataLoader(
            TorchCIFAR100Fed("test", transform),
            batch_size=64,
            num_workers=self.config.DL_N_WORKER,
        )

        return train_loader_list, test_loader

    def get_iid_data(self, transform):
        logging.info("Torch CIFAR100 loading ...")
        cifar100_train_ds = datasets.CIFAR10(
            "data/cifar10", download=True, transform=transform,
        )
        logging.info("Torch CIFAR100 loaded")
        logging.info("IID distribution ...")
        n_training_samples = len(cifar100_train_ds)
        indices = np.arange(n_training_samples)
        indices = indices.reshape(self.config.N_CLIENTS, -1)
        indices = indices.tolist()

        train_loader_list = []
        for idx in indices:
            sampler = th.utils.data.sampler.SubsetRandomSampler(idx)
            loader = th.utils.data.DataLoader(
                dataset=cifar100_train_ds,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.DL_N_WORKER,
                sampler=sampler,
            )
            train_loader_list.append(loader)
        logging.info("IID distributed")
        return train_loader_list

    def get_non_iid_data(self, transform):
        logging.info("Non IID loading ...")
        clients = [str(x) for x in np.arange(self.N_TRAINING_CLIENTS)]
        indices = np.array_split(clients, self.config.N_CLIENTS)
        train_loader_list = []
        for indice in indices:
            ds = TorchCIFAR100Fed(indice, transform)
            loader = th.utils.data.DataLoader(
                dataset=ds,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.DL_N_WORKER,
            )
            train_loader_list.append(loader)
        logging.info("Non IID loaded")
        return train_loader_list

    def get_model_cls(self) -> Callable[[], nn.Module]:
        return Net

    def get_loss(self):
        return nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
