from comet_ml import Experiment
import torch as th

import logging
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig


def get_args(opt):
    if opt == "Adam":
        return {"betas": (0.0, 0.99), "eps": 0.01}
    elif opt == "SGD":
        return {"momentum": 0.9}
    else:
        return {}


def do_training(config: TorchFederatedLearnerCIFAR100Config, config_technical: TorchFederatedLearnerTechnicalConfig):
    name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config, config_technical)
    learner.train()


th.cuda.set_device(0)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

project_name = "adam_init"

max_rounds = 200
C = 10 / 500
NC = 500
E = 1
B = 20
is_iid = False
server_lr = 1.0
client_lr = 0.1
server_opt = "SGD"
client_opt = "SGD"
client_opt_strategy = "reinit"
# image_norm = "tflike"
# TODO a paraméterek helytelen nevére nem adott hibát
config = TorchFederatedLearnerCIFAR100Config(
    BREAK_ROUND=300,
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=client_opt,
    CLIENT_OPT_ARGS=get_args(client_opt),
    # CLIENT_OPT_L2=1e-4,
    CLIENT_OPT_STRATEGY=client_opt_strategy,
    SERVER_OPT=server_opt,
    # SERVER_OPT_ARGS=get_args(server_opt),
    SERVER_LEARNING_RATE=server_lr,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=max_rounds,
    DL_N_WORKER=0,
    NORM="group",
    # IMAGE_NORM=image_norm,
    INIT="tffed",  # "keras",
)
config_technical = TorchFederatedLearnerTechnicalConfig()

do_training(config, config_technical)