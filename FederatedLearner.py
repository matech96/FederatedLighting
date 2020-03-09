from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Callable
import logging

import tensorflow as tf
import tensorflow_federated as tff
from comet_ml.exceptions import InterruptedExperiment

from comet_ml import Experiment


def factory(c, **kwarg):
    return staticmethod(lambda: c(**kwarg))


@dataclass
class FederatedLearnerConfig:
    CLIENT_OPT_FN: Callable = factory(tf.keras.optimizers.SGD, learning_rate=0.02)
    SERVER_OPT_FN: Callable = factory(tf.keras.optimizers.SGD, learning_rate=1.0)
    N_ROUNDS: int = 20  # The number of round for training (analogous for number of epochs).
    N_CLIENTS: int = 10  # The number of clients to participate in a round.
    TEST_AFTER: int = 10  # Evaluate on the test set after every TEST_AFTER epoch.


class FederatedLearner(ABC):
    def __init__(self, experiment: Experiment, config: FederatedLearnerConfig) -> None:
        """
        Initialises the training.
        :param experiment: Comet.ml experiment object for online logging.
        """
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.experiment.log_parameters(self.config.__dict__)

    @abstractmethod
    def load_data(self) -> Tuple[List, List]:  # BatchDataset
        """
        Loads the data to BatchDataset. The BatchDataset should contain and OrderedDict([(x, tf.Tensor(
        dtype=tf.float32)), (y, tf.Tensor(tf.int32))]) @return: BatchDataset
        """
        pass

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """
        Builds a keras Model without compiling it.
        @return: not compiled keras Model
        """
        pass

    @abstractmethod
    def get_loss(self) -> tf.keras.losses.Loss:
        """
        Returns the training loss
        @return: training loss
        """
        pass

    def train(self) -> None:
        """
        Runs the federated training, reports to comet.ml and runs an evaluation at the end.
        """

        federated_train_data, federated_test_data = self.load_data()
        sample_batch = tf.nest.map_structure(
            lambda x: x.numpy(), next(iter(federated_train_data[0]))
        )

        def model_fn():
            keras_model = self.build_model()
            loss = self.get_loss()
            return tff.learning.from_keras_model(keras_model, sample_batch, loss)

        logging.info("Initialization ...")
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=self.config.CLIENT_OPT_FN,
            server_optimizer_fn=self.config.SERVER_OPT_FN,
        )

        state = iterative_process.initialize()
        logging.info("Initialized")
        try:
            for round_num in range(self.config.N_ROUNDS):
                state, metrics = iterative_process.next(state, federated_train_data)
                self.log_mterics_in_round(metrics, round_num, "train")

                if round_num % self.config.TEST_AFTER == 0:
                    evaluation = tff.learning.build_federated_evaluation(model_fn)
                    test_metrics = evaluation(state.model, federated_test_data)
                    self.log_mterics_in_round(test_metrics, round_num, "test")
        except InterruptedExperiment:
            pass

    def log_mterics_in_round(self, metrics, round_num, prefix):
        for name, value in metrics._asdict().items():
            self.experiment.log_metric(f"{prefix}_{name}", value, step=round_num)
