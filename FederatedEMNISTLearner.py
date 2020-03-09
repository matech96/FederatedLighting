import logging
import collections
import functools
from typing import List, Tuple
from dataclasses import dataclass

import tensorflow_federated as tff
import tensorflow as tf

from FederatedLearner import FederatedLearner, FederatedLearnerConfig
from comet_ml import Experiment


@dataclass
class FederatedEMNISTLearnerConfig(FederatedLearnerConfig):
    BATCH_SIZE: int = 20
    SHUFFLE_BUFFER: int = 500
    IS_PLOT_DATA: bool = False
    WRITER_LIMIT: int = None  # How many writer to include in the data. If None, all writers are included.


class FederatedEMNISTLearner(FederatedLearner):
    def __init__(
        self, experiment: Experiment, config: FederatedEMNISTLearnerConfig
    ) -> None:
        """
        Initialises the training.
        :param experiment: Comet.ml experiment object for online logging.
        """
        super().__init__(experiment, config)
        self.config = config  # Purly to help intellisense

    def load_data(self) -> Tuple[List, List]:  # [BatchDataset]
        logging.info("dataset loading ...")
        (emnist_train, emnist_test,) = tff.simulation.datasets.emnist.load_data()

        def preprocess(dataset):
            def element_fn(element):
                return collections.OrderedDict(
                    [
                        ("x", tf.reshape(element["pixels"], [-1])),
                        ("y", tf.reshape(element["label"], [1])),
                    ]
                )

            return (
                dataset.repeat(self.config.N_ROUNDS)
                .map(element_fn)
                .shuffle(self.config.SHUFFLE_BUFFER)
                .batch(self.config.BATCH_SIZE)
            )

        def get_federated_data(dataset):
            federated_data = [list() for _ in range(self.config.N_CLIENTS)]
            for i, client_id in enumerate(dataset.client_ids):
                client_data = dataset.create_tf_dataset_for_client(client_id)
                federated_data[i % self.config.N_CLIENTS].append(client_data)
                if (self.config.WRITER_LIMIT is not None) and (
                    i == self.config.WRITER_LIMIT
                ):
                    break
            return [
                preprocess(functools.reduce(lambda a, b: a.concatenate(b), fd))
                for fd in federated_data
            ]

        federated_train_data = get_federated_data(emnist_train)
        federated_test_data = get_federated_data(emnist_test)
        if self.config.IS_PLOT_DATA:
            self.plot_data_first_batch(federated_train_data)

        logging.info("dataset loaded")
        return federated_train_data, federated_test_data

    def build_model(self) -> tf.keras.Model:
        logging.info("Model building ...")
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(10, kernel_initializer="zeros"),
                tf.keras.layers.Softmax(),
            ]
        )
        logging.info("Model built")

        return model

    def get_loss(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.SparseCategoricalCrossentropy()

    def plot_data_first_batch(self, federated_train_data):
        sample_batch = tf.nest.map_structure(
            lambda x: x.numpy(), next(iter(federated_train_data[0]))
        )
        for i in range(sample_batch["y"].shape[0]):
            self.experiment.log_image(
                sample_batch["x"][i,].reshape((28, 28)), name=sample_batch["y"][i]
            )
