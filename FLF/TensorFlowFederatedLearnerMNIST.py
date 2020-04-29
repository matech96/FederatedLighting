from comet_ml import Experiment

from typing import Tuple, List, Callable
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense

from FLF.TensorFlowFederatedLearner import (
    TensorFlowFederatedLearner,
    TensorFlowFederatedLearnerConfig,
)


class TensorFlowFederatedLearnerMNISTConfig(TensorFlowFederatedLearnerConfig):
    IS_IID_DATA: bool = True  # If true, the data is split random amongs clients. If false, the client have different digits.


class TensorFlowFederatedLearnerMNIST(TensorFlowFederatedLearner):
    def __init__(
        self, experiment: Experiment, config: TensorFlowFederatedLearnerMNISTConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {TensorFlowFederatedLearnerMNISTConfig} -- Training configuration description.
        """
        super().__init__(experiment, config)
        self.config = config  # Purly to help intellisense

    def load_data(self,) -> Tuple[List[tf.data.Dataset], tf.data.Dataset]:
        logging.info("MNIST data loading ...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        logging.info("MNIST data loaded.")

        logging.info("Data for client is being sampled ...")
        n_training_samples = len(y_train)
        logging.info("Number of training samples: {n_training_samples}")
        if self.config.IS_IID_DATA:
            indices = np.arange(n_training_samples)
            indices = indices.reshape(self.config.N_CLIENTS, -1)
            indices = indices.tolist()
        else:
            indices = self.__distribute_data_non_IID(y_train)

        train_loader_list = []
        for idx in indices:
            np.random.shuffle(idx)
            loader = (
                tf.data.Dataset.from_tensor_slices((x_train[idx], y_train[idx]))
                .map(to_format)
                .cache()
                .batch(self.config.BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            train_loader_list.append(loader)
        logging.info("Data for client is sampled.")

        test_loader = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .map(to_format)
            .cache()
            .batch(64)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return train_loader_list, test_loader

    def __distribute_data_non_IID(self, y_train):
        digit_sort_idx = np.concatenate([np.where(y_train == i)[0] for i in range(10)])
        digit_sort_idx = digit_sort_idx.reshape(2 * self.config.N_CLIENTS, -1)
        np.random.shuffle(digit_sort_idx)
        indices = [
            digit_sort_idx[i : i + 2, ].flatten()
            for i in range(0, 2 * self.config.N_CLIENTS, 2)
        ]
        return indices

    def get_model_cls(self) -> Callable[[], tf.keras.Model]:
        def model_builder():
            model = tf.keras.Sequential(
                [
                    InputLayer((28, 28, 1)),
                    Conv2D(32, 5, activation="relu"),
                    Conv2D(64, 5, activation="relu"),
                    Flatten(),
                    Dense(512, activation="relu"),
                    Dense(10),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.SGD(
                    learning_rate=self.config.LEARNING_RATE
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
            return model

        return model_builder


def to_format(x, y):
    return (tf.cast(tf.expand_dims(x, -1), tf.float32) / 255, y)
