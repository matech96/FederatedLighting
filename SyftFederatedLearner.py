from comet_ml.exceptions import InterruptedExperiment
from comet_ml import Experiment

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict
import logging

import syft as sy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


@dataclass
class SyftFederatedLearnerConfig:
    N_ROUNDS: int = 10  # The number of round for training (analogous for number of epochs).
    TEST_AFTER: int = 1  # Evaluate on the test set after every TEST_AFTER rounds.
    N_CLIENTS: int = 2  # The number of clients to participate in a round.
    # N_EPOCH_PER_CLIENT: int = 1  # The number of epoch to train on the client before
    BATCH_SIZE: int = 64  # Batch size
    LEARNING_RATE: float = 0.01  # Learning rate for the local optimizer
    DL_N_WORKER: int = 4  # Syft.FederatedDataLoader: number of workers
    # LOG_INTERVALL_STEP: int = 30  # The client reports it's performance to comet.ml after every LOG_INTERVALL_STEP update in the round.


class SyftFederatedLearner:
    def __init__(
        self, experiment: Experiment, config: SyftFederatedLearnerConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {SyftFederatedLearnerConfig} -- Training configuration description.
        """
        super().__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.config = config
        self.experiment.log_parameters(self.config.__dict__)
        self.hook = sy.TorchHook(th)
        self.clients = [
            sy.VirtualWorker(self.hook) for _ in range(self.config.N_CLIENTS)
        ]

    @abstractmethod
    def load_data(self) -> Tuple[sy.FederatedDataLoader, th.utils.data.DataLoader]:
        """Loads the data.

        Returns:
            Tuple[sy.FederatedDataLoader, th.utils.data.DataLoader] -- [The first element is the training set, the second is the test set]
        """
        pass

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Returns the model to be trained.

        Returns:
            nn.Module -- The instance of the model.
        """
        pass

    def train(self) -> None:
        """Runs the federated training, reports to comet.ml and runs an evaluations.

        Returns:
            None -- No return value.
        """
        federated_train_loader, test_loader = self.load_data()
        n_train_batches = len(federated_train_loader)

        model = self.build_model().to(self.device)
        optimizer = optim.SGD(
            model.parameters(), lr=self.config.LEARNING_RATE
        )  # TODO momentum is not supported at the moment

        for round in range(self.config.N_ROUNDS):
            self.__train_one_round(model, federated_train_loader, optimizer, round, n_train_batches)
            metrics = self.test(model, test_loader)
            self.log_test_metric(metrics, round, n_train_batches)

        # th.save(model.state_dict(), "mnist_cnn.pt")

    def __train_one_round(self, model, federated_train_loader, optimizer, round, n_batches):
        model.train()
        for batch_num, (data, target) in enumerate(federated_train_loader):
            model.send(data.location)

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            model.get()
            loss = loss.get()

            self.log_client_step(
                loss.item(), data.location.id, (round * n_batches) + batch_num
            )

    def test(
        self, model: nn.Module, test_loader: th.utils.data.DataLoader
    ) -> Dict[str, float]:
        model.eval()
        test_loss = 0
        correct = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100.0 * correct / len(test_loader.dataset)
        return {"test_loss": test_loss, "test_acc": test_acc}

    def log_client_step(self, loss, client_id, batch_num):
        self.experiment.log_metric(f"{client_id}_train_loss", loss, step=batch_num)

    def log_test_metric(self, metrics, round, n_train_batches):
        for name, value in metrics.items():
            self.experiment.log_metric(name, value, step=round * n_train_batches)
