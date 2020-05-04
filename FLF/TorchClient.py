import logging
from typing import Callable, Dict

import torch as th
import torch.nn as nn

from FLF.TorchModelOptStateManager import TorchModelOptStateManager


class TorchClient:
    __next_ID = 0

    @classmethod
    def reset_ID_counter(cls):
        cls.__next_ID = 0

    def __init__(
        self,
        trainer,
        model_cls: Callable[[], th.nn.Module],
        is_keep_model_on_gpu: bool,
        loss: nn.Module,
        dataloader: th.utils.data.DataLoader,
        device: str,
        opt_cls: Callable[..., th.optim.Optimizer],
        opt_cls_param: Dict,
        is_maintaine_opt_state: bool,
    ):
        self.id = TorchClient.__next_ID
        TorchClient.__next_ID += 1

        self.trainer = trainer
        self.loss = loss
        self.dataloader = dataloader
        self.device = device
        self.state_man = TorchModelOptStateManager(model_cls, opt_cls, opt_cls_param, is_keep_model_on_gpu, self.id)
        self.is_maintaine_opt_state = is_maintaine_opt_state

        self.opt = None

        logging.info(f"Client {self.id} was created")

    def set_model(
        self, model_state_dict
    ):  # TODO Doc: you have to call this before train_round!
        self.state_man.set_model_state(model_state_dict)

    def set_opt_state(self, state):
        self.state_man.set_opt_state(state)

    def train_round(
        self, n_epochs, curr_round
    ):  # TODO DOC: curr_round for logging purpuses.
        model, opt = self.state_man.instanciate()
        for curr_epoch in range(n_epochs):
            for curr_batch, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                opt.zero_grad()
                output = model(data)
                loss = self.loss(output, target)
                loss.backward()
                opt.step()

                if (curr_batch == 0) or (curr_batch % 10 == 0):
                    self.trainer.log_client_step(
                        loss.item(), self.id, curr_round, curr_epoch, curr_batch
                    )

        model_state = model.state_dict()
        opt_state = opt.state_dict()["state"].values()
        if self.is_maintaine_opt_state:
            self.state_man.set_opt_state(opt_state)

        return model_state, opt_state

    def __log(self, m):
        logging.info(f"Client {self.id}: {m}")
