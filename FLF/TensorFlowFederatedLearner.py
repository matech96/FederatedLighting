from comet_ml.exceptions import InterruptedExperiment
from comet_ml import Experiment

import random
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Callable
import logging

import numpy as np
from pydantic import BaseModel, validator

import tensorflow as tf

from FLF.TensorFlowClient import TensorFlowClient


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
    SEED: int = None  # The seed.

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
        if config.SEED is not None:  # TODO test deterministic
            random.seed(config.SEED)
            np.random.seed(config.SEED)
            tf.random.set_seed(config.SEED)

        self.experiment = experiment
        self.config = config
        self.experiment.log_parameters(self.config.__dict__)

        model_cls = self.get_model_cls()
        self.model = model_cls()

        self.train_loader_list, self.test_loader = self.load_data()
        self.n_train_batches = int(
            len(self.train_loader_list) / self.config.N_CLIENTS
        )  # TODO batch per client
        logging.info(f"Number of training batches: {self.n_train_batches}")

        self.clients = [
            TensorFlowClient(self, model_cls, loader)
            for loader in self.train_loader_list
        ]

    @abstractmethod
    def load_data(self,) -> Tuple[List[tf.data.Dataset], tf.data.Dataset]:
        """Loads the data.

        Returns:
            Tuple[List[tf.data.Dataset], tf.data.Dataset] -- [The first element is the training set, the second is the test set] 
        """
        pass

    @abstractmethod
    def get_model_cls(self) -> Callable[[], tf.keras.Model]:
        """Returns the model to be trained.

        Returns:
            tf.keras.Model -- The instance of the model.
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

                if metrics["sparse_categorical_accuracy"] > self.config.TARGET_ACC:
                    break
        except InterruptedExperiment:
            pass

        # th.save(model.state_dict(), "mnist_cnn.pt")

    def __train_one_round(self, curr_round: int):
        client_sample = self.__select_clients()
        for client in client_sample:
            client.set_model(self.model.get_weights())

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
        final_state_dict = np.mean(collected_model_state_dicts, axis=0)
        self.model.set_weights(final_state_dict)

    def test(self, test_loader: tf.data.Dataset) -> Dict[str, float]:
        result = self.model.evaluate(test_loader)
        return dict(zip(self.model.metrics_names, result))

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
            nice_value = 100 * value if name.endswith("_accuracy") else value
            self.experiment.log_metric("round_" + name, nice_value, step=batch_num)
