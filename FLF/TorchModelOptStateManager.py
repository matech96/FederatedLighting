from pathlib import Path
import logging

import torch as th
import os

os.makedirs("tmp", exist_ok=True)


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

        self.__model_state_to_be_loaded = None
        self.__opt_state_to_be_loaded = None
        self.__opt_path = self.tmp_dir / f"{self.id}_opt.pt"
        self.__delete_objects_tmp_files()

        self.model = None
        self.opt = None

    def __del__(self):
        self.__delete_objects_tmp_files()

    def get_current_model_state(self):
        return self.model.state_dict()

    def get_current_opt_state(self):
        return self.opt.state_dict()["state"].values()

    def set_model_state_to_be_loaded(self, state):
        self.__model_state_to_be_loaded = state
        self.__log("model set")

    def set_opt_state_to_be_loaded(self, state, is_preserve=False):
        if is_preserve and self.is_store_on_disk:
            th.save(list(state), self.__opt_path)
            self.__log(f"opt saved: {self.__opt_path}")
        else:
            self.__opt_state_to_be_loaded = state
            self.__log("opt set")

    def __enter__(self):
        if self.model is None:
            self.model = self.model_cls()
            self.__log("model instanciated")
        if self.__model_state_to_be_loaded is not None:
            self.model.load_state_dict(self.__model_state_to_be_loaded)
            self.__log("model state loaded")
        self.model.cuda()
        self.__log("model is on GPU")

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

    def __delete_objects_tmp_files(self):
        if self.__opt_path.exists():
            self.__opt_path.unlink()

    def __log(self, m):
        logging.info(f"Client {self.id}: {m}")
