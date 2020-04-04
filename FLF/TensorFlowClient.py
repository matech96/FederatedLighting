from typing import Callable

import tensorflow as tf

from FLF.TensorFlowFederatedLearner import TensorFlowFederatedLearnerConfig


class TensorFlowClient:
    __next_ID = 0

    def __init__(
        self,
        trainer,
        model_cls: Callable[[], tf.keras.Model],
        dataloader: tf.data.Dataset,
        config: TensorFlowFederatedLearnerConfig,
    ):
        self.id = TensorFlowClient.__next_ID
        TensorFlowClient.__next_ID += 1

        self.trainer = trainer
        self.model = model_cls()
        self.dataloader = dataloader

    def set_model(self, weights):  # TODO Doc: you have to call this before train_round!
        self.model.set_weights(weights)

    def train_round(
        self, n_epochs, curr_round
    ):  # TODO DOC: curr_round for logging purpuses.
        self.model.fit(self.dataloader, epochs=n_epochs)  # TODO wandb callback

    def get_model_state_dict(self):
        return self.model.get_weights()
