import collections
from typing import List

import tensorflow_federated as tff
import tensorflow as tf

from FederatedLearner import FederatedLearner


class FederatedMNISTLearner(FederatedLearner):
    def load_data(self) -> List:  # [BatchDataset]
        (
            emnist_train,
            emnist_test,
        ) = tff.simulation.datasets.emnist.load_data()  # TODO use test dataset
        # TODO dynamicly set these
        NUM_CLIENTS = 10
        NUM_EPOCHS = 10
        BATCH_SIZE = 20
        SHUFFLE_BUFFER = 500

        def preprocess(dataset):
            def element_fn(element):
                return collections.OrderedDict(
                    [
                        ("x", tf.reshape(element["pixels"], [-1])),
                        ("y", tf.reshape(element["label"], [1])),
                    ]
                )

            return (
                dataset.repeat(NUM_EPOCHS)
                .map(element_fn)
                .shuffle(SHUFFLE_BUFFER)
                .batch(BATCH_SIZE)
            )

        def get_federated_data(dataset):
            return [
                preprocess(dataset.create_tf_dataset_for_client(x))
                for x in emnist_train.client_ids
            ]

        federated_train_data = get_federated_data(emnist_train)
        self.plot_data_first_batch(federated_train_data)
        federated_test_data = get_federated_data(emnist_test)

        return federated_train_data, federated_test_data

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(10, kernel_initializer="zeros"),
                tf.keras.layers.Softmax(),
            ]
        )

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
