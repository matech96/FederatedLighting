from comet_ml.exceptions import InterruptedExperiment
from comet_ml import Experiment

import random
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Callable
import logging

import numpy as np
from pydantic import BaseModel, validator

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from syftutils.multipointer import avg_model_state_dicts

from FLF.TorchOptRepo import TorchOptRepo
from FLF.TorchClient import TorchClient


class TensorFlowFederatedLearnerConfig(BaseModel):
    class Config:
        validate_assignment = True

    TARGET_ACC: float = 0.99  # The training stopps when the test accuracy is higher, than this value.
    MAX_ROUNDS: int = 10  # The maximum number of round for training.
    N_CLIENTS: int = 2  # The number of clients to participate in a round.
    CLIENT_FRACTION: float = 1.0  # The fration of clients to participate in 1 round. Muss be between 0 and 1. 0 means selecting 1 client.
    N_EPOCH_PER_CLIENT: int = 1  # The number of epoch to train on the client before sync.
    BATCH_SIZE: int = 64  # Batch size. If set to sys.maxsize, the epoch is processed in a single batch.
    LEARNING_RATE: float = 0.01  # Learning rate for the local optimizer
    DL_N_WORKER: int = 4  # Syft.FederatedDataLoader: number of workers
    SEED: int = None  # The seed.
    OPT: str = "SGD"  # The optimizer used by the client.

    @staticmethod
    def __percentage_validator(value: float) -> None:
        if (0.0 > value) or (value > 1.0):
            raise ValueError("CLIENT_FRACTION muss be between 0 and 1.")
        else:
            return value

    _val_CLIENT_FRACTION = validator("CLIENT_FRACTION", allow_reuse=True)(
        __percentage_validator.__func__
    )
    _val_TARGET_ACC = validator("TARGET_ACC", allow_reuse=True)(
        __percentage_validator.__func__
    )


class TensorFlowFederatedLearner(ABC):
    def __init__(
        self, experiment: Experiment, config: TensorFlowFederatedLearnerConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {TorchFederatedLearnerConfig} -- Training configuration description.
        """
        super().__init__()
        if config.SEED is not None:
            random.seed(config.SEED)
            np.random.seed(config.SEED)
            th.manual_seed(config.SEED)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False

        self.device = "cuda"  # th.device("cuda" if th.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.config = config
        self.experiment.log_parameters(self.config.__dict__)

        model_cls = self.get_model_cls()
        self.model = model_cls().to(self.device)

        self.train_loader_list, self.test_loader = self.load_data()
        self.n_train_batches = int(
            len(self.train_loader_list) / self.config.N_CLIENTS
        )  # TODO batch per client
        logging.info(f"Number of training batches: {self.n_train_batches}")

        TorchClient.reset_ID_counter()
        self.clients = [
            TorchClient(
                self,
                model_cls,
                loader,
                self.device,
                TorchOptRepo.name2cls(self.config.OPT),
                {"lr": self.config.LEARNING_RATE},
            )
            for loader in self.train_loader_list
        ]

    @abstractmethod
    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader]:
        """Loads the data.

        Returns:
            Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader] -- [The first element is the training set, the second is the test set]
        """
        pass

    @abstractmethod
    def get_model_cls(self) -> Callable[[], nn.Module]:
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
            for round in range(self.config.MAX_ROUNDS):
                self.experiment.log_parameter("curr_round", round)
                self.__train_one_round(round)
                metrics = self.test(self.test_loader)
                self.log_test_metric(
                    metrics,
                    round * self.config.N_EPOCH_PER_CLIENT * self.n_train_batches,
                )

                if metrics["test_acc"] > self.config.TARGET_ACC:
                    break
        except InterruptedExperiment:
            pass

        # th.save(model.state_dict(), "mnist_cnn.pt")

    def __train_one_round(self, curr_round: int):
        self.model.train()

        client_sample = self.__select_clients()
        for client in client_sample:
            client.set_model(self.model.state_dict())

        for client in client_sample:
            client.train_round(self.config.N_EPOCH_PER_CLIENT, curr_round)

        self.__collect_avg_model(client_sample)

    def __select_clients(self):
        client_sample = random.sample(
            self.clients, max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        )
        logging.info(f"Selected {len(client_sample)} clients in this round.")
        return client_sample

    def __collect_avg_model(self, client_sample):
        collected_model_state_dicts = [
            client.get_model_state_dict() for client in client_sample
        ]
        final_state_dict = avg_model_state_dicts(collected_model_state_dicts)
        self.model.load_state_dict(final_state_dict)

    def test(self, test_loader: th.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        test_loss = 0
        correct = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        return {"test_loss": test_loss, "test_acc": test_acc}

    def log_client_step(
        self,
        loss: float,
        client_id: str,
        curr_round: int,
        curr_epoch: int,
        curr_batch: int,
    ):
        step = (
            (curr_round * self.config.N_EPOCH_PER_CLIENT) + curr_epoch
        ) * self.n_train_batches + curr_batch
        logging.info(
            f"R: {curr_round:4} E: {curr_epoch:4} B: {curr_batch:4} (S: {step} C: {client_id}) Training loss: {loss}"
        )
        self.experiment.log_metric(f"{client_id}_train_loss", loss, step=step)

    def log_test_metric(self, metrics: Dict[str, float], batch_num: int):
        self.experiment.log_parameter(
            "TOTAL_EPOCH",
            self.config.N_EPOCH_PER_CLIENT
            * batch_num
            / (self.config.N_EPOCH_PER_CLIENT * self.n_train_batches),
        )
        for name, value in metrics.items():
            nice_value = 100 * value if name.endswith("_acc") else value
            self.experiment.log_metric(name, nice_value, step=batch_num)
