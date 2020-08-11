from comet_ml import Experiment

import logging
from typing import Tuple, List, Callable

import numpy as np
import torch as th
import torch.nn as nn
from torchvision import datasets, transforms

from FLF.TorchFederatedLearner import TorchFederatedLearner, TorchFederatedLearnerConfig
from FLF.data.TorchCIFAR100Fed import TorchCIFAR100Fed
from FLF.model.TorchResNetFactory import TorchResNetFactory


class TorchFederatedLearnerCIFAR100Config(TorchFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.
    IMAGE_NORM: str = "thlike"  # The way to normalize the images. Options: "tflike", "thlike"
    NORM: str = "batch"  # Normalization layer of ResNet. Options: "batch", "group"
    INIT: str = None  # Initialization of ResNet weights. Options: "keras"


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
        if self.config.IMAGE_NORM == "thlike":
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.config.IMAGE_NORM == "tflike":
            norm = transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        else:
            raise Exception("IMAGE_NORM not supported!")

        trfs = [
            transforms.ToTensor(),
            norm,
        ]

        transform = transforms.Compose(trfs)

        if self.config.IS_IID_DATA:
            train_loader_list = self.get_iid_data(transform)
        else:
            train_loader_list = self.get_non_iid_data(transform)

        test_loader = th.utils.data.DataLoader(
            TorchCIFAR100Fed("test", transform),
            batch_size=64,
            num_workers=self.config.DL_N_WORKER,
        )

        return train_loader_list, test_loader, 0.01

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
                pin_memory=True,
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
                pin_memory=True,
            )
            train_loader_list.append(loader)
        logging.info("Non IID loaded")
        return train_loader_list

    def get_model_cls(self) -> Callable[[], nn.Module]:
        return TorchResNetFactory(self.config.NORM, self.config.INIT)

    def get_loss(self):
        return nn.CrossEntropyLoss()
