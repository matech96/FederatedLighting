from typing import Callable, Dict

import torch as th
import torch.nn.functional as F


class TorchClient:
    __next_ID = 0

    @classmethod
    def reset_ID_counter(cls):
        cls.__next_ID = 0

    def __init__(
        self,
        trainer,
        model_cls: Callable[[], th.nn.Module],
        dataloader: th.utils.data.DataLoader,
        device: str,
        opt_cls: Callable[..., th.optim.Optimizer],
        opt_cls_param: Dict,
        is_maintaine_opt_state: bool
    ):
        self.id = TorchClient.__next_ID
        TorchClient.__next_ID += 1

        self.trainer = trainer
        self.model = model_cls()
        self.dataloader = dataloader
        self.device = device
        self.opt_cls = opt_cls
        self.opt_cls_param = opt_cls_param
        self.is_maintaine_opt_state = is_maintaine_opt_state

        self.opt = None

    def set_model(
        self, model_state_dict
    ):  # TODO Doc: you have to call this before train_round!
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        if (self.opt is not None) and self.is_maintaine_opt_state:
            old_state = self.get_opt_state()
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_cls_param)
            self.set_opt_state(old_state)
        else:
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_cls_param)

    def set_opt_state(self, state):
        new_state_dict = self.opt.state_dict()
        new_state_dict['state'].update(zip(new_state_dict['param_groups'][0]['params'], state))
        self.opt.load_state_dict(new_state_dict)

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

                if (curr_batch == 1) or (curr_batch % 10 == 0):
                    self.trainer.log_client_step(
                        loss.item(), self.id, curr_round, curr_epoch, curr_batch
                    )

    def get_opt_state(self):
        return self.opt.state_dict()['state'].values()
    
    def get_model_state_dict(self):
        return self.model.state_dict()
