from pathlib import Path
import logging
from typing import Callable, Dict

import torch as th
import copy
import os

from FLF.TorchOptRepo import TorchOptRepo

os.makedirs("tmp", exist_ok=True)


class TorchModelOptStateManager:
    tmp_dir = Path("tmp")

    def __init__(
        self,
        model_cls: Callable[[], th.nn.Module],
        opt_cls: Callable[..., th.optim.Optimizer],
        opt_cls_param: Dict,
        is_keep_model_on_gpu: bool,
        is_store_on_disk: bool,
        id: int,
        exp_id: str,
        device
    ):
        self.device = device
        self.model_cls = model_cls
        self.opt_cls = opt_cls
        self.opt_cls_param = opt_cls_param
        self.is_keep_model_on_gpu = is_keep_model_on_gpu
        self.is_store_on_disk = is_store_on_disk
        self.id = id

        self.__model_state_to_be_loaded = None
        self.__opt_state_to_be_loaded = None

        self.__opt_path = self.tmp_dir / f"{exp_id}_{self.id}_opt.pt"
        self.__c_path = self.tmp_dir / f"{exp_id}_{self.id}_c.pt"
        self.__delete_objects_tmp_files()

        self.model = None
        self.opt = None
        self.c = None

    def __del__(self):
        self.__delete_objects_tmp_files()

    def switch_to_sgd(self, lr):
        # TODO __del__
        self.__delete_objects_tmp_files()
        # opt_cls = sgd
        self.opt_cls = TorchOptRepo.name2cls("SGD")
        # delete self.opt
        self.opt = None
        # delete __opt_state_to_be_loaded
        self.__opt_state_to_be_loaded = None
        # lr
        self.opt_cls_param = {
            "lr": lr,
            "weight_decay": self.opt_cls_param["weight_decay"],
        }

    def get_current_model_state(self):
        return copy.deepcopy(self.model.state_dict())

    def get_current_opt_state(self):
        return self.opt.state_dict()["state"].values()

    def set_model_state_to_be_loaded(self, state):
        self.__model_state_to_be_loaded = copy.deepcopy(state)
        self.__log("model set")

    def set_opt_state_to_be_loaded(self, state, is_preserve=False):
        if is_preserve and self.is_store_on_disk:
            th.save(list(state), self.__opt_path)
            self.__log(f"opt saved: {self.__opt_path}")
        else:
            self.__opt_state_to_be_loaded = list(state)  # TODO state is probably on gpu
            self.__log("opt set")

    def __enter__(self):
        if self.model is None:
            self.model = self.model_cls()
            self.__log("model instanciated")
        if self.__model_state_to_be_loaded is not None:
            self.model.load_state_dict(self.__model_state_to_be_loaded)
            self.__log("model state loaded")
        self.model.train()
        self.model.to(self.device)
        self.__log(f"model is on {self.device}")

        if self.opt is None:
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_cls_param)
            self.__log("opt instanciated")
        opt_state = None
        if self.__opt_state_to_be_loaded is not None:
            opt_state = self.__opt_state_to_be_loaded
            self.__log("using preset opt state")
        elif self.__opt_path.exists():
            opt_state = th.load(self.__opt_path)
            self.__log(f"opt read from disk: {self.__opt_path}")
        if opt_state is not None:
            new_state_dict = self.opt.state_dict()
            new_state_dict["state"].update(
                zip(new_state_dict["param_groups"][0]["params"], opt_state,)
            )
            self.opt.load_state_dict(new_state_dict)
            self.__log("opt state loaded")

        if self.__c_path.exists():
            assert self.c is None
            self.c = th.load(self.__c_path)
            self.__log(f"c read from disk: {self.__c_path}")

    def __exit__(self, *exc):
        if not self.is_keep_model_on_gpu:
            self.model.cpu()
            self.opt = None
            self.__log("model is on CPU")

        if self.is_store_on_disk:
            self.opt = None
            self.model = None
            self.__opt_state_to_be_loaded = None
            self.__model_state_to_be_loaded = None

            if self.is_store_on_disk and (self.c is not None):
                th.save(list(self.c), self.__c_path)
                self.__log(f"c saved: {self.__c_path}")
            self.c = None

    def __delete_objects_tmp_files(self):
        if self.__opt_path.exists():
            self.__opt_path.unlink()

        if self.__c_path.exists():
            self.__c_path.unlink()

    def __log(self, m):
        logging.info(f"Client {self.id}: {m}")
