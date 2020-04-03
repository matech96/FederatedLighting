from typing import Callable

import torch as th
import torch.nn.functional as F


class TensorFlowClient:
    __next_ID = 0

    def __init__(
        self,
        trainer,
        model_cls: Callable[[], th.nn.Module],
        dataloader: th.utils.data.DataLoader,
        device,
    ):
        self.id = TensorFlowClient.__next_ID
        TensorFlowClient.__next_ID += 1

        self.trainer = trainer
        self.model = model_cls()
        self.dataloader = dataloader
        self.device = device

    def set_model(
        self, model_state_dict, config
    ):  # TODO Doc: you have to call this before train_round!
        self.model.load_state_dict(model_state_dict)  # TODO tf.keras.models.clone_model
        self.model.to(self.device)  # TODO remove
        self.opt = th.optim.SGD(
            self.model.parameters(), lr=config.LEARNING_RATE
        )  # TODO keras
        # TODO compile

    def train_round(
        self, n_epochs, curr_round
    ):  # TODO DOC: curr_round for logging purpuses.
        # TODO remove the hole funtion. use keras api instead
        for curr_epoch in range(n_epochs):
            for curr_batch, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.opt.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.opt.step()

                if (curr_batch == 1) or (curr_batch % 10 == 0):
                    self.trainer.log_client_step(
                        loss.item(), self.id, curr_round, curr_epoch, curr_batch
                    )

    def get_model_state_dict(self):
        return self.model.state_dict()  # TODO remove function
