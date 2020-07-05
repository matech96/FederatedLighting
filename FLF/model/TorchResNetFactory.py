from torch import nn
import torchvision.models as models


class TorchResNetFactory:
    def __init__(self, norm="batch", init=None):
        self.norm = norm
        self.init = init

    def __call__(self):
        if self.norm == "batch":
            model = models.resnet18()
        elif self.config.NORM == "group":
            make_group_norm = lambda x: nn.GroupNorm(2, x)
            model = models.resnet18(norm_layer=make_group_norm)
        else:
            raise Exception("NORM is not supported!")

        if self.init is not None:
            if self.init == "keras":
                model = model.apply(self.__keras_like_init)
            else:
                raise Exception("INIT is not supported!")
        return model

    def __keras_like_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
