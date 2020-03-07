from abc import ABC, abstractmethod
from typing import List

import collections

import tensorflow as tf
import tensorflow_federated as tff
from comet_ml.exceptions import InterruptedExperiment

# from tensorflow.python.data.ops.dataset_ops import BatchDataset

from comet_ml import Experiment


class FederatedLearner(ABC):
    def __init__(self, experiment: Experiment) -> None:
        """
        Initialises the training.
        :param experiment: Comet.ml experiment object for online logging.
        """
        super().__init__()
        self.experiment = experiment

    @abstractmethod
    def load_data(self) -> List:  # BatchDataset
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

    def train(self, n_rounds: int, client_sample_percent: float) -> None:
        """
        Runs the federated training, reports to comet.ml and runs an evaluation at the end.
        @param n_rounds: The number of round for training (analogous for number of epochs).
        @param client_sample_percent: The percentage of clients to participate in a round. Muss be between 0 and 1.
        """

        # emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        # print("In[6]:")

        # example_dataset = emnist_train.create_tf_dataset_for_client(
        #     emnist_train.client_ids[0]
        # )

        # print("In[8]:")

        # NUM_CLIENTS = 10
        # NUM_EPOCHS = 10
        # BATCH_SIZE = 20
        # SHUFFLE_BUFFER = 500

        # def preprocess(dataset):
        #     def element_fn(element):
        #         return collections.OrderedDict(
        #             [
        #                 ("x", tf.reshape(element["pixels"], [-1])),
        #                 ("y", tf.reshape(element["label"], [1])),
        #             ]
        #         )

        #     return (
        #         dataset.repeat(NUM_EPOCHS)
        #         .map(element_fn)
        #         .shuffle(SHUFFLE_BUFFER)
        #         .batch(BATCH_SIZE)
        #     )

        # # Let's verify this worked.

        # print("In[9]:")

        # preprocessed_example_dataset = preprocess(example_dataset)

        # sample_batch = tf.nest.map_structure(
        #     lambda x: x.numpy(), iter(preprocessed_example_dataset).next()
        # )

        # print("In[10]:")

        # def make_federated_data(client_data, client_ids):
        #     return [
        #         preprocess(client_data.create_tf_dataset_for_client(x))
        #         for x in client_ids
        #     ]

        # print("In[11]:")

        # sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

        # federated_train_data = make_federated_data(emnist_train, sample_clients)

        # len(federated_train_data), federated_train_data[0]

        print("In[12]:")

        # next(iter(federated_train_data[0]))

        federated_train_data = self.load_data()
        sample_batch = tf.nest.map_structure(
            lambda x: x.numpy(), next(iter(federated_train_data[0]))
        )
        # ## Creating a model with Keras
        #
        # If you are using Keras, you likely already have code that constructs a Keras
        # model. Here's an example of a simple model that will suffice for our needs.

        print("In[13]:")

        # def create_compiled_keras_model():
        #     model = tf.keras.models.Sequential(
        #         [
        #             tf.keras.layers.Dense(
        #                 10,
        #                 activation=tf.nn.softmax,
        #                 kernel_initializer="zeros",
        #                 input_shape=(784,),
        #             )
        #         ]
        #     )

        #     model.compile(
        #         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #         optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        #         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        #     )
        #     return model

        # print("In[14]:")

        def model_fn():
            keras_model = self.build_model()
            loss = self.get_loss()
            return tff.learning.from_keras_model(keras_model, sample_batch, loss)

        print("In[15]:")

        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        )
        # def get_model_fun(keras_model, sample_batch):
        #     return lambda: tff.learning.from_compiled_keras_model(keras_model, sample_batch)

        # iterative_process = tff.learning.build_federated_averaging_process(get_model_fun(self.build_model(), sample_batch))

        # print("In[17]:")

        # state = iterative_process.initialize()

        # print("In[18]:")

        # state, metrics = iterative_process.next(state, federated_train_data)
        # print("round  1, metrics={}".format(metrics))

        # print("In[19]:")

        # NUM_ROUNDS = 3
        # for round_num in range(2, NUM_ROUNDS):
        #     state, metrics = iterative_process.next(state, federated_train_data)
        #     print("round {:2d}, metrics={}".format(round_num, metrics))

        # federated_train_data = self.load_data()
        # emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
        # example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
        # # sample_batch = next(iter(federated_train_data[0]))
        # NUM_CLIENTS = 10
        # NUM_EPOCHS = 10
        # BATCH_SIZE = 20
        # SHUFFLE_BUFFER = 500

        # def preprocess(dataset):

        #     def element_fn(element):
        #         return collections.OrderedDict([
        #             ('x', tf.reshape(element['pixels'], [-1])),
        #             ('y', tf.reshape(element['label'], [1])),
        #         ])

        #     return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        #         SHUFFLE_BUFFER).batch(BATCH_SIZE)
        # preprocessed_example_dataset = preprocess(example_dataset)

        # sample_batch = tf.nest.map_structure(
        #     lambda x: x.numpy(), iter(preprocessed_example_dataset).next())
        # print("sample_batch")

        # def create_compiled_keras_model():
        #     model = tf.keras.models.Sequential([
        #         tf.keras.layers.Dense(
        #             10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

        #     model.compile(
        #         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #         optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        #         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        #     return model
        # def model_fn():
        #     keras_model = create_compiled_keras_model()
        #     return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

        # iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        # print("iterative_process")

        # # def get_model_fun(keras_model, loss, sample_batch):
        # #     return lambda: tff.learning.from_keras_model(keras_model, sample_batch, loss)

        # # print(tf.executing_eagerly())
        # # iterative_process = tff.learning.build_federated_averaging_process(get_model_fun(self.build_model(), self.get_loss(), sample_batch))

        state = iterative_process.initialize()
        try:
            for round_num in range(n_rounds):
                print(f"Round: {round_num}")
                state, metrics = iterative_process.next(state, federated_train_data)
                for value, name in zip(metrics, dir(metrics)):
                    self.experiment.log_metric(name, value, step=round_num)
        except InterruptedExperiment:
            pass
