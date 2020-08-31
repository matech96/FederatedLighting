from torch import nn
import torchvision.models as models

import logging


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
            if self.init == "keras":
                init_fn = self.__keras_like_init
                logging.info("ResNet keras initialized")
            elif self.init == "tffed":
                init_fn = self.__tffed_like_init
                logging.info("ResNet tffed initialized")
            elif self.init == "fcdebug":
                init_fn = self.__fcdebug_init
                logging.info("ResNet fcdebug initialized")
            else:
                raise Exception("INIT is not supported!")
            model = model.apply(init_fn)
        return model

    def __keras_like_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def __tffed_like_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            nn.init.normal_(m.weight, std=0.01)

    def __fcdebug_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.bias, std=0.03)
            nn.init.normal_(m.weight, std=0.03)
