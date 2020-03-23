from comet_ml import Experiment

from typing import Callable
import logging

import torch as th
import torch.nn.functional as F


class Client:
    __next_ID = 0

    def __init__(
        self,
        trainer,
        model_cls: Callable[[], th.nn.Module],
        dataloader: th.utils.data.DataLoader,
        device,
    ):
        self.id = Client.__next_ID
        Client.__next_ID += 1

        self.trainer = trainer
        self.model = model_cls()
        self.dataloader = dataloader
        self.device = device

    def set_model(
        self, model_state_dict, config
    ):  # TODO Doc: you have to call this before train_round!
        self.model.load_state_dict(model_state_dict)
        self.opt = th.optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE)

    def train_round(
        self, n_epochs, curr_round
    ):  # TODO DOC: curr_round for logging purpuses.
        for curr_epoch in range(n_epochs):
            for curr_batch, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.opt.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.opt.step()

                loss = loss.get()

                self.trainer.log_client_step(
                    loss.item(), self.id, curr_round, curr_epoch, curr_batch
                )

    def get_model_state_dict(self):
        return self.model.state_dict()
