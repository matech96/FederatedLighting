import logging


class TorchModelOptStateManager:
    def __init__(self, model_cls, opt_cls, opt_cls_param, is_keep_model_on_gpu, id):
        self.model_cls = model_cls
        self.opt_cls = opt_cls
        self.opt_cls_param = opt_cls_param
        self.is_keep_model_on_gpu = is_keep_model_on_gpu
        self.id = id

        self.model_state = None
        self.opt_state = None

        self.model = None
        self.opt = None

    @property
    def model_state(self):
        return self.model.state_dict()

    @model_state.setter
    def model_state(self, state):
        self.model_state = state
        self.__log("model set")

    @property
    def opt_state(self):
        return self.opt.state_dict()["state"].values()

    @opt_state.setter
    def opt_state(self, state):
        self.opt_state = state
        self.__log("opt set")

    def __enter__(self):
        if self.model is None:
            self.model = self.model_cls()
            self.__log("model instanciated")
        if self.model_state is not None:
            self.model.load_state_dict(self.model_state)
            self.__log("model state loaded")
        self.model.cuda()
        self.__log("model is on GPU")

        if self.opt is None:
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_cls_param)
            self.__log("opt instanciated")
        if self.opt_state is not None:
            new_state_dict = self.opt.state_dict()
            new_state_dict["state"].update(
                zip(new_state_dict["param_groups"][0]["params"], self.opt_state)
            )
            self.opt.load_state_dict(new_state_dict)
            self.__log("opt state loaded")

    def __exit__(self, *exc):
        if not self.is_keep_model_on_gpu:
            self.model.cpu()
            self.__log("model is on CPU")

    def __log(self, m):
        logging.info(f"Client {self.id}: {m}")
