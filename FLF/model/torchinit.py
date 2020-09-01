import math
import torch as th
from torch import nn
import inspect
from typing import List

from FLF.model import torchinit


def keras(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)


def tffed(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.normal_(m.weight, std=0.01)


def fcdebug(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.bias, std=0.03)
        nn.init.normal_(m.weight, std=0.03)


def book(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight.data, a=0, mode="fan_out", nonlinearity="relu",
        )
        if m.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(m.bias, -bound, bound)


def xavier_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(m.weight)


def xavier_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(m.weight)


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="conv2d")
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


def kaiming_normal_fan_out(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="conv2d")
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


def kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="conv2d")
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")


def kaiming_uniform_fan_out(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="conv2d")
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")


def normal(m):
    nn.init.normal_(m.weight)


def uniform(m):
    nn.init.uniform_(m.weight)


class TorchInitRepo:
    repo = {
        f[0]: f[1] for f in inspect.getmembers(torchinit) if inspect.isfunction(f[1])
    }

    @classmethod
    def get_opt_names(cls) -> List:
        res = list(cls.repo.keys())
        return res

    @classmethod
    def name2fn(cls, name) -> th.optim.Optimizer:
        return cls.repo[name]
