from typing import List, Tuple, Callable, Dict
import logging

import syft as sy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from SyftFederatedLearner import SyftFederatedLearner


class SyftFederatedLearnerMNIST(SyftFederatedLearner):
    def load_data(self) -> Tuple[sy.FederatedDataLoader, th.utils.data.DataLoader]:
        logging.info('Train MNIST data loading ...')
        federated_train_loader = sy.FederatedDataLoader(
            datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ).federate(self.clients),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.DL_N_WORKER,
            pin_memory=True,
        )
        logging.info('Train MNIST data loaded.')

        logging.info('Test MNIST data loading ...')
        test_loader = th.utils.data.DataLoader(
            datasets.MNIST(
                "../data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.DL_N_WORKER,
            pin_memory=True,
        )
        logging.info('Test MNIST data loaded.')

        return federated_train_loader, test_loader

    def build_model(self) -> nn.Module:
        return Net()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
