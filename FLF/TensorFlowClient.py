from typing import Callable

import tensorflow as tf


class TensorFlowClient:
    __next_ID = 0

    def __init__(
        self,
        trainer,
        model_cls: Callable[[], tf.keras.Model],
        dataloader: tf.data.Dataset,
        # device,
    ):
        self.id = TensorFlowClient.__next_ID
        TensorFlowClient.__next_ID += 1

        self.trainer = trainer
        self.model = model_cls()
        self.dataloader = dataloader
        # self.device = device

    def set_model(
        self, weights, config
    ):  # TODO Doc: you have to call this before train_round!
        # self.model.load_state_dict(model)
        # self.model = tf.keras.models.clone_model(model)
        self.model.set_weights(weights)
        # self.model.to(self.device)
        # self.opt = th.optim.SGD(
        #     self.model.parameters(), lr=config.LEARNING_RATE
        # )
        self.opt = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE)
        self.model.compile(
            optimizer=self.opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    def train_round(
        self, n_epochs, curr_round
    ):  # TODO DOC: curr_round for logging purpuses.
        # TODO remove the hole funtion. use keras api instead
        self.model.fit(self.dataloader, epochs=n_epochs)  # TODO wandb callback
        # for curr_epoch in range(n_epochs):
        #     for curr_batch, (data, target) in enumerate(self.dataloader):
        #         data, target = data.to(self.device), target.to(self.device)
        #         self.opt.zero_grad()
        #         output = self.model(data)
        #         loss = F.nll_loss(output, target)
        #         loss.backward()
        #         self.opt.step()

        #         if (curr_batch == 1) or (curr_batch % 10 == 0):
        #             self.trainer.log_client_step(
        #                 loss.item(), self.id, curr_round, curr_epoch, curr_batch
        #             )

    def get_model_state_dict(self):
        return self.model.get_weights()
        # return self.model.state_dict()  # TODO remove function
