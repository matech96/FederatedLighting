import logging
from typing import Callable, List, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from comet_ml import Experiment
from torchvision import datasets, transforms

from FLF.data.TorchCIFAR100Fed import TorchCIFAR100Fed
from FLF.model.TorchResNetFactory import TorchResNetFactory
from FLF.TorchFederatedLearner import (
    TorchFederatedLearner,
    TorchFederatedLearnerConfig,
    TorchFederatedLearnerTechnicalConfig,
)


class TorchFederatedLearnerCIFAR100Config(TorchFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.
    IMAGE_NORM: str = "thlike"  # The way to normalize the images. Options: "tflike", "thlike", "recordwise"
    NORM: str = "batch"  # Normalization layer of ResNet. Options: "batch", "group"
    INIT: str = None  # Initialization of ResNet weights. Options: None, "keras", "tffed", "fcdebug"
    AUG: str = None  # Data augmentation. Options: None, "basic"
    SHUFFLE: str = True  # Data shuffeling


class TorchFederatedLearnerCIFAR100(TorchFederatedLearner):
    N_TRAINING_CLIENTS = 500

    def __init__(
        self,
        experiment: Experiment,
        config: TorchFederatedLearnerCIFAR100Config,
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
        train_transform, test_transform = self.__get_transformations()

        if self.config.IS_IID_DATA:
            train_loader_list = self.get_iid_data(train_transform)
        else:
            train_loader_list = self.get_non_iid_data(train_transform)

        test_loader = th.utils.data.DataLoader(
            TorchCIFAR100Fed("test", test_transform),
            batch_size=64,
            num_workers=self.config_technical.DL_N_WORKER,
        )

        return train_loader_list, test_loader, 0.01

    def __get_transformations(self):
        if self.config.IMAGE_NORM == "thlike":
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.config.IMAGE_NORM == "tflike":
            norm = transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        elif self.config.IMAGE_NORM == "recordwise":
            norm = transforms.Lambda(
                lambda x: transforms.functional.normalize(
                    x, x.view(3, -1).mean(dim=1), x.view(3, -1).std(dim=1)
                )
            )
        else:
            raise Exception("IMAGE_NORM not supported!")

        trfs = [
            transforms.ToTensor(),
            norm,
        ]
        if self.config.AUG is not None:
            if self.config.AUG == "basic":
                train_trfs = [
                    transforms.ToPILImage(),
                    transforms.RandomCrop(24),
                    transforms.RandomHorizontalFlip(),
                ] + trfs
                test_trfs = [transforms.ToPILImage(), transforms.CenterCrop(24)] + trfs
            elif self.config.AUG == "24":
                train_trfs = [transforms.ToPILImage(), transforms.CenterCrop(24)] + trfs
                test_trfs = train_trfs
            else:
                raise Exception("AUG not supported!")
        else:
            train_trfs = trfs
            test_trfs = trfs

        return transforms.Compose(train_trfs), transforms.Compose(test_trfs)

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
                num_workers=self.config_technical.DL_N_WORKER,
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
                shuffle=True,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config_technical.DL_N_WORKER,
                pin_memory=True,
            )
            train_loader_list.append(loader)
        logging.info("Non IID loaded")
        return train_loader_list

    def get_model_cls(self) -> Callable[[], nn.Module]:
        return TorchResNetFactory(self.config.NORM, self.config.INIT)

    def get_loss(self, **kwargs):
        return nn.CrossEntropyLoss(**kwargs)
