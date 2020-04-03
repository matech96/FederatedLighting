from comet_ml import Experiment

from typing import Tuple, List, Callable
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense

# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms

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

    def load_data(
        self,
    ) -> Tuple[List[tf.data.Dataset], tf.data.Dataset]:  # TODO datatype
        # TODO assert only 2 clients are supported
        logging.info("MNIST data loading ...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # minist_train_ds, mnist_test_ds = self.__get_mnist()  # TODO load from keras
        logging.info("MNIST data loaded.")

        logging.info("Data for client is being sampled ...")
        n_training_samples = len(y_train)  # TODO use x_train instead
        logging.info("Number of training samples: {n_training_samples}")
        if self.config.IS_IID_DATA:
            indices = np.arange(n_training_samples)
            indices = indices.reshape(self.config.N_CLIENTS, -1)
            indices = indices.tolist()
        else:
            indices = self.__distribute_data_non_IID(y_train)

        to_format = lambda x, y: (tf.cast(tf.expand_dims(x, -1), tf.float32) / 255, y)
        train_loader_list = []
        for idx in indices:
            np.random.shuffle(idx)
            loader = (
                tf.data.Dataset.from_tensor_slices((x_train[idx], y_train[idx]))
                .map(to_format)
                .batch(self.config.BATCH_SIZE)
            )
            # sampler = th.utils.data.sampler.SubsetRandomSampler(
            #     idx
            # )  # TODO numpy slicing on x_train and y_train
            # sampler = th.utils.data.sampler.SequentialSampler(idx)
            # loader = th.utils.data.DataLoader(  # TODO tf.DataSet # TODO apply normalization
            #     dataset=minist_train_ds,
            #     batch_size=self.config.BATCH_SIZE,
            #     num_workers=self.config.DL_N_WORKER,
            #     sampler=sampler,
            # )
            train_loader_list.append(loader)
        logging.info("Data for client is sampled.")

        test_loader = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .map(to_format)
            .batch(64)
        )
        # test_loader = th.utils.data.DataLoader(  # TODO tf.Dataset # TODO apply normalization
        #     mnist_test_ds, batch_size=64, num_workers=self.config.DL_N_WORKER,
        # )

        return train_loader_list, test_loader

    # def __get_mnist(self):  # TODO remove
    #     minist_train_ds = datasets.MNIST(
    #         "../data",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     )
    #     mnist_test_ds = datasets.MNIST(
    #         "../data",
    #         train=False,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     )
    #     return minist_train_ds, mnist_test_ds

    def __distribute_data_non_IID(self, y_train):  # TODO Take y_train
        digit_sort_idx = np.concatenate(
            [
                np.where(y_train == i)[0] for i in range(10)
            ]  # TODO instead of minist_train_ds.targets use y_train
        )
        digit_sort_idx = digit_sort_idx.reshape(2 * self.config.N_CLIENTS, -1)
        np.random.shuffle(digit_sort_idx)
        indices = [
            digit_sort_idx[i : i + 2,].flatten()
            for i in range(0, 2 * self.config.N_CLIENTS, 2)
        ]
        return indices

    def get_model_cls(self) -> Callable[[], tf.keras.Model]:
        return lambda: tf.keras.Sequential(
            [
                InputLayer((28, 28, 1)),
                Conv2D(32, 5, activation="relu"),
                Conv2D(64, 5, activation="relu"),
                Flatten(),
                Dense(512, activation="relu"),
                Dense(10),
            ]
        )
