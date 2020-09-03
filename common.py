from comet_ml import Experiment
import logging

from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args(opt):
    if (opt == "Adam") or (opt == "Yogi"):
        return {"betas": (0.0, 0.99), "eps": 0.01}
    elif opt == "SGD":
        return {"momentum": 0.9}
    else:
        return {}


def do_training(
    name,
    project_name,
    config: TorchFederatedLearnerCIFAR100Config,
    config_technical: TorchFederatedLearnerTechnicalConfig,
):
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config, config_technical)
    learner.train()
