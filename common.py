import numpy as np
import pandas as pd
import comet_ml
from comet_ml.exceptions import NotFound
import logging
import datetime as dt

from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.TorchFederatedLearnerEMNIST import (
    TorchFederatedLearnerEMNIST,
    TorchFederatedLearnerEMNISTConfig,
)
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig


# logging.basicConfig(
#     format="%(asctime)s %(levelname)-8s %(message)s",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


def get_args(opt):
    if opt == "Adam":
        return {"betas": (0.9, 0.999), "eps": 0.001}
    elif opt == "SGDM":
        return {"momentum": 0.9}
    else:
        return {}


def get_name(opt):
    if opt == "SGDM":
        return "SGD"
    else:
        return opt


def do_training(
    experiment,
    name,
    config: TorchFederatedLearnerCIFAR100Config,
    config_technical: TorchFederatedLearnerTechnicalConfig,
):
    logging.info(name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config, config_technical)
    learner.train()


def do_training_emnist(
    experiment,
    name,
    config: TorchFederatedLearnerEMNISTConfig,
    config_technical: TorchFederatedLearnerTechnicalConfig,
):
    logging.info(name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerEMNIST(experiment, config, config_technical)
    learner.train()


def get_besr_lrs_from_exps(SOPT, STR="r", COPT="sgd"):
    comet_api = comet_ml.api.API()
    maxes = pd.DataFrame(columns=["E", "slr", "clr"])
    for E in [1, 5, 10, 20, 30]:
        try:
            exps = comet_api.get(
                f"federated-learning-emnist-s/cnn3400c{E}e100r10f-{SOPT}-{STR}-{COPT}"
            )
        except NotFound:
            break
        slr = exp_params2list(exps, "SERVER_LEARNING_RATE", float)
        clr = exp_params2list(exps, "CLIENT_LEARNING_RATE", float)
        acc = exp_metrics2list(exps, "last_avg_acc", float)
        df = pd.DataFrame({"slr": np.log10(slr), "clr": np.log10(clr), "acc": acc})
        i = df["acc"].idxmax()
        m = df.iloc[i]
        maxes = maxes.append(
            {"E": E, "slr": m["slr"], "clr": m["clr"]}, ignore_index=True
        )
    return maxes
