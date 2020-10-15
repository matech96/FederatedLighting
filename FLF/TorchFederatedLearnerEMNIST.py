import os
from typing import Tuple, List, Callable
import logging
from comet_ml import Experiment

from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import tensorflow_federated as tff

from FLF.data.TorchEMNISTFed import TorchEMNISTFed
from FLF.TorchFederatedLearner import (
    TorchFederatedLearner,
    TorchFederatedLearnerConfig,
    TorchFederatedLearnerTechnicalConfig,
)


class TorchFederatedLearnerEMNISTConfig(TorchFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.
    SHUFFLE: str = True  # Data shuffeling
    MODEL: str = "2NN"  # Type of the neural network. Options: "2NN", "CNN"


class TorchFederatedLearnerEMNIST(TorchFederatedLearner):
    def __init__(
        self,
        experiment: Experiment,
        config: TorchFederatedLearnerEMNISTConfig,
        config_technical: TorchFederatedLearnerTechnicalConfig,
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {TorchFederatedLearnerCIFAR100Config} -- Training configuration description.
        """
        super().__init__(experiment, config, config_technical)
        self.config = config  # Purly to help intellisense

    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader, float]:
        transform = None

        if self.config.IS_IID_DATA:
            train_loader_list = self.get_iid_data(transform)
        else:
            train_loader_list = self.get_non_iid_data(transform)

        test_loader = th.utils.data.DataLoader(
            TorchEMNISTFed("test", transform),
            batch_size=64,
            num_workers=self.config_technical.DL_N_WORKER,
        )

        random_acc = 1 / len(np.unique(test_loader.dataset.labels))
        return train_loader_list, test_loader, random_acc

    def get_iid_data(self, transform):
        logging.info("Torch EMNIST loading ...")
        ids = TorchEMNISTFed.get_client_ids("train")
        train_ds = TorchEMNISTFed(ids)
        logging.info("Torch EMNIST loaded")
        logging.info("IID distribution ...")
        n_training_samples = len(train_ds)
        indices = np.arange(n_training_samples)
        indices = indices.reshape(self.config.N_CLIENTS, -1)
        indices = indices.tolist()

        train_loader_list = []
        for idx in indices:
            sampler = th.utils.data.sampler.SubsetRandomSampler(idx)
            loader = th.utils.data.DataLoader(
                dataset=train_ds,
                shuffle=self.config.SHUFFLE,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config_technical.DL_N_WORKER,
                pin_memory=self.config_technical.PIN_MEMORY,
                sampler=sampler,
            )
            train_loader_list.append(loader)
        logging.info("IID distributed")
        return train_loader_list

    def get_non_iid_data(self, transform):
        logging.info("Non IID loading ...")
        clients = TorchEMNISTFed.get_client_ids("train")
        indices = np.array_split(clients, self.config.N_CLIENTS)
        train_loader_list = []
        for indice in indices:
            ds = TorchEMNISTFed(indice, transform)
            loader = th.utils.data.DataLoader(
                dataset=ds,
                shuffle=self.config.SHUFFLE,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config_technical.DL_N_WORKER,
                pin_memory=self.config_technical.PIN_MEMORY,
            )
            train_loader_list.append(loader)
        logging.info("Non IID loaded")
        return train_loader_list

    def get_model_cls(self) -> Callable[[], nn.Module]:
        if self.config.MODEL == "2NN":
            return Net2NN
        elif self.config.MODEL == "CNN":
            return NetCNN

    def get_loss(self, **kwargs):
        return nn.CrossEntropyLoss(**kwargs)


class NetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 62)
        self.dropout = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 62)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
