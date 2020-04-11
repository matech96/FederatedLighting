from typing import List, Union

import torch as th


class TorchOptRepo:
    repo = {x.__name__: x for x in th.optim.Optimizer.__subclasses__()}

    @classmethod
    def get_opt_names(cls) -> List:
        res = list(cls.repo.keys())
        res.remove("SparseAdam")
        res.remove("LBFGS")
        return res

    @classmethod
    def name2cls(cls, name) -> th.optim.Optimizer:
        return cls.repo[name]

    @classmethod
    def supported_parameters(cls, opt: Union[str, th.optim.Optimizer]) -> List[str]:
        if isinstance(opt, str):
            opt_ = cls.name2cls(opt)
        else:
            opt_ = opt

        res = list(opt_.__init__.__code__.co_varnames)
        res.remove("defaults")
        res.remove("self")
        res.remove("params")
        return res
