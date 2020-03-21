from comet_ml.exceptions import InterruptedExperiment
from comet_ml import Experiment

import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict
import logging

from pydantic import BaseModel, validator

import syft as sy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from syftutils.multipointer import avg_models


class SyftFederatedLearnerConfig(BaseModel):
    class Config:
        validate_assignment = True

    N_ROUNDS: int = 10  # The number of round for training (analogous for number of epochs).
    TEST_AFTER: int = 1  # Evaluate on the test set after every TEST_AFTER rounds.
    N_CLIENTS: int = 2  # The number of clients to participate in a round.
    CLIENT_FRACTION: float = 1.0  # The fration of clients to participate in 1 round. Muss be between 0 and 1. 0 means selecting 1 client.
    N_EPOCH_PER_CLIENT: int = 1  # The number of epoch to train on the client before sync.
    BATCH_SIZE: int = 64  # Batch size
    LEARNING_RATE: float = 0.01  # Learning rate for the local optimizer
    DL_N_WORKER: int = 4  # Syft.FederatedDataLoader: number of workers
    # LOG_INTERVALL_STEP: int = 30  # The client reports it's performance to comet.ml after every LOG_INTERVALL_STEP update in the round.

    @validator("CLIENT_FRACTION", check_fields=False)  # class method
    def CLIENT_FRACTION_muss_be_procentage(cls, value: float) -> None:
        if (0.0 > value) or (value > 1.0):
            raise ValueError("CLIENT_FRACTION muss be between 0 and 1.")
        else:
            return value


class SyftFederatedLearner(ABC):
    def __init__(
        self, experiment: Experiment, config: SyftFederatedLearnerConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {SyftFederatedLearnerConfig} -- Training configuration description.
        """
        super().__init__()
        self.device = "cpu"  # th.device("cuda" if th.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.config = config
        self.experiment.log_parameters(self.config.__dict__)
        self.hook = sy.TorchHook(th)
        self.clients = [
            sy.VirtualWorker(self.hook, f"{i}") for i in range(self.config.N_CLIENTS)
        ]

        self.federated_train_loader, self.test_loader = self.load_data()
        self.n_train_batches = int(
            len(self.federated_train_loader) / self.config.N_CLIENTS
        )  # TODO batch per client
        logging.info(f"Number of training batches: {self.n_train_batches}")

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
        try:
            model = self.build_model().to(self.device)
            for round in range(self.config.N_ROUNDS):
                model = self.__train_one_round(model, round)
                metrics = self.test(model, self.test_loader)
                self.log_test_metric(
                    metrics,
                    round * self.config.N_EPOCH_PER_CLIENT * self.n_train_batches,
                )
        except InterruptedExperiment:
            pass

        # th.save(model.state_dict(), "mnist_cnn.pt")

    def __train_one_round(self, model: nn.Module, round: int):
        model.train()

        client_sample = self.__select_clients()
        optimizer_ptrs, model_ptrs = self.__send_model_to_clients(model, client_sample)

        for epoch_num in range(self.config.N_EPOCH_PER_CLIENT):
            model = self.__train_one_epoch(optimizer_ptrs, model_ptrs, round, epoch_num)

        model = self.__collect_avg_model(model_ptrs)
        return model

    def __select_clients(self):
        client_sample = random.sample(
            self.clients, max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        )
        logging.info(f"Selected {len(client_sample)} clients in this round.")
        return client_sample

    def __train_one_epoch(self, optimizer_ptrs, model_ptrs, curr_round, curr_epoch):
        for curr_batch, (data, target) in enumerate(self.federated_train_loader):
            client_id = data.location.id
            if client_id not in model_ptrs.keys():
                continue

            optimizer = optimizer_ptrs[client_id]
            model = model_ptrs[client_id]

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            loss = loss.get()

            self.log_client_step(
                loss.item(), data.location.id, curr_round, curr_epoch, curr_batch
            )
        return model

    def __send_model_to_clients(
        self, model: nn.Module, client_sample
    ) -> Tuple[Dict, Dict]:
        model_ptrs = {client.id: model.copy().send(client) for client in client_sample}
        optimizer_ptrs = {
            client.id: optim.SGD(
                model_ptrs[client.id].parameters(), lr=self.config.LEARNING_RATE
            )
            for client in client_sample
        }  # TODO momentum is not supported at the moment
        return optimizer_ptrs, model_ptrs

    def __collect_avg_model(self, model_ptrs):
        collected_models = [model.get() for model in model_ptrs.values()]
        model = avg_models(collected_models)
        return model

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

    def log_client_step(
        self,
        loss: float,
        client_id: str,
        curr_round: int,
        curr_epoch: int,
        curr_batch: int,
    ):
        if (curr_batch % 10) != 0:
            return

        step = (
            (curr_round * self.config.N_EPOCH_PER_CLIENT) + curr_epoch
        ) * self.n_train_batches + curr_batch
        logging.info(
            f"R: {curr_round:4} E: {curr_epoch:4} B: {curr_batch:4} (S: {step} C: {client_id}) Training loss: {loss}"
        )
        self.experiment.log_metric(f"{client_id}_train_loss", loss, step=step)

    def log_test_metric(self, metrics: Dict, batch_num: int):
        for name, value in metrics.items():
            self.experiment.log_metric(name, value, step=batch_num)
