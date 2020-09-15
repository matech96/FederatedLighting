import logging
from typing import Callable, Dict
import copy

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
        is_store_opt_on_disk: bool,
        loss: nn.Module,
        dataloader: th.utils.data.DataLoader,
        device: str,
        opt_cls: Callable[..., th.optim.Optimizer],
        opt_cls_param: Dict,
        is_maintaine_opt_state: bool,
        exp_id: str,
    ):
        self.id = TorchClient.__next_ID
        TorchClient.__next_ID += 1

        self.trainer = trainer
        self.loss = loss
        self.dataloader = dataloader
        self.device = device
        self.state_man = TorchModelOptStateManager(
            model_cls,
            opt_cls,
            opt_cls_param,
            is_keep_model_on_gpu,
            is_store_opt_on_disk,
            self.id,
            exp_id,
        )
        self.is_maintaine_opt_state = is_maintaine_opt_state

        self.opt = None

        logging.info(f"Client {self.id} was created")

    def set_model(
        self, model_state_dict
    ):  # TODO Doc: you have to call this before train_round!
        self.state_man.set_model_state_to_be_loaded(copy.deepcopy(model_state_dict))

    def set_opt_state(self, state):
        self.state_man.set_opt_state_to_be_loaded(state)

    def train_round(
        self, n_epochs, curr_round
    ):  # TODO DOC: curr_round for logging purpuses.
        with self.state_man:
            for curr_epoch in range(n_epochs):
                correct = 0
                for curr_batch, (data, target) in enumerate(self.dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.state_man.opt.zero_grad()
                    output = self.state_man.model(data)
                    loss = self.loss(output, target)
                    loss.backward()
                    self.state_man.opt.step()

                    pred = output.argmax(
                        1, keepdim=True
                    )  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    if (curr_batch == 0) or (curr_batch % 10 == 0):
                        self.trainer.log_client_step(
                            loss.item(), self.id, curr_round, curr_epoch, curr_batch
                        )
                train_acc = correct / len(self.dataloader.dataset)

            self.trainer.log_metric(
                {f"{self.id}_train_acc": train_acc},
                curr_round,
            )

            if self.is_maintaine_opt_state:
                self.state_man.set_opt_state_to_be_loaded(
                    self.state_man.get_current_opt_state(), True
                )

            return (
                self.state_man.get_current_model_state(),
                self.state_man.get_current_opt_state(),
            )

    def __log(self, m):
        logging.info(f"Client {self.id}: {m}")
