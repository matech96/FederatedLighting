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

from FLF.data import TorchEMNISTFed
from FLF.TorchFederatedLearner import TorchFederatedLearner, TorchFederatedLearnerConfig, TorchFederatedLearnerTechnicalConfig


class TorchFederatedLearnerEMNISTConfig(TorchFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.
    INIT: str = None  # Initialization of ResNet weights. Options: None, "keras", "tffed", "fcdebug"
    SHUFFLE: str = True  # Data shuffeling


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
        transform = transforms.ToTensor()

        if self.config.IS_IID_DATA:
            train_loader_list = self.get_iid_data(
                transform
            )
        else:
            train_loader_list = self.get_non_iid_data(
                transform
            )

        test_loader = th.utils.data.DataLoader(
            TorchEMNISTFed("test", transform),
            batch_size=64,
            num_workers=self.config_technical.DL_N_WORKER,
        )

        return train_loader_list, test_loader, 0.1
        
    def get_iid_data(self, transform):
        logging.info("Torch EMNIST loading ...")
        train_ds = datasets.EMNIST(
            "data/emnist", "digits", download=True, transform=transform,
        )
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
        return Net
        
    def get_loss(self, **kwargs):
        return nn.CrossEntropyLoss(**kwargs)


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
