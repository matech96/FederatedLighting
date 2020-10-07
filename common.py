import logging

from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.TorchFederatedLearnerEMNIST import (
    TorchFederatedLearnerEMNIST,
    TorchFederatedLearnerEMNISTConfig,
)
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
