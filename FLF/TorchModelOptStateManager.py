from pathlib import Path
import logging

import torch as th
import os
import shutil

from mutil.pickle import save, load

os.makedirs('tmp', exist_ok=True)
shutil.rmtree("tmp")
os.mkdir('tmp')


class TorchModelOptStateManager:
    tmp_dir = Path("tmp")

    def __init__(
        self,
        model_cls,
        opt_cls,
        opt_cls_param,
        is_keep_model_on_gpu,
        is_store_on_disk,
        id,
    ):
        self.model_cls = model_cls
        self.opt_cls = opt_cls
        self.opt_cls_param = opt_cls_param
        self.is_keep_model_on_gpu = is_keep_model_on_gpu
        self.is_store_on_disk = is_store_on_disk
        self.id = id

        self.__model_path = self.tmp_dir / f"{self.id}_model.pt"
        self.__opt_path = self.tmp_dir / f"{self.id}_opt.pt"

        self.model = None
        self.opt = None

    def get_current_model_state(self):
        return self.model.state_dict()

    def get_current_opt_state(self):
        return self.opt.state_dict()["state"].values()

    def set_model_state_to_be_loaded(self, state):
        th.save(state, self.__model_path)
        self.__log("model saved")

    def set_opt_state_to_be_loaded(self, state):
        save(state, self.__opt_path)
        self.__log("opt saved")

    def __enter__(self):
        if self.model is None:
            self.model = self.model_cls()
            self.__log("model instanciated")
        if self.__model_path.exists():
            model_state = th.load(self.__model_path)
            self.model.load_state_dict(model_state)
            self.__log("model state loaded")
        self.model.cuda()
        self.__log("model is on GPU")

        if self.opt is None:
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_cls_param)
            self.__log("opt instanciated")
        if self.__opt_path.exists():
            try:
                new_state_dict = self.opt.state_dict()
                new_state_dict["state"].update(
                    zip(
                        new_state_dict["param_groups"][0]["params"],
                        load(self.__opt_path),
                    )
                )
                self.opt.load_state_dict(new_state_dict)
                self.__log("opt state loaded")
            except Exception as e:
                print(th.load(self.__opt_path))
                raise e

    def __exit__(self, *exc):
        if not self.is_keep_model_on_gpu:
            self.model.cpu()
            self.__log("model is on CPU")

        if self.is_store_on_disk:
            self.opt = None
            self.model = None

    def __log(self, m):
        logging.info(f"Client {self.id}: {m}")
