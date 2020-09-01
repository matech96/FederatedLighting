from torch import nn
import torchvision.models as models

import logging

from FLF.model.torchinit import TorchInitRepo


class TorchResNetFactory:
    def __init__(self, norm="batch", init=None):
        self.norm = norm
        self.init = init

    def __call__(self):
        kwargs = {"num_classes": 100}
        if self.norm == "batch":
            logging.info("ResNet keras batchnorm")
        elif self.norm == "group":
            kwargs["norm_layer"] = lambda x: nn.GroupNorm(2, x)
            logging.info("ResNet keras groupnorm")
        else:
            raise Exception("NORM is not supported!")
        model = models.resnet18(**kwargs)

        if self.init is not None:
            init_fn = TorchInitRepo.name2fn(self.init)
            model = model.apply(init_fn)
        return model
