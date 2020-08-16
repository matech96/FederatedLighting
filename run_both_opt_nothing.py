from comet_ml import Experiment
import torch as th

import logging
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.BreakedTrainingExcpetion import ToLargeLearningRateExcpetion


def get_args(opt):
    if opt == "Adam":
        return {"betas": (0.0, 0.99), "eps": 0.01}
    elif opt == "SGD":
        return {"momentum": 0.9}
    else:
        return {}


def do_training(config: TorchFederatedLearnerCIFAR100Config):
    name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config)
    learner.train()


# th.cuda.set_device(2)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

project_name = "speed-meassure"

max_rounds = 1500
C = 1# 10 / 500
NC = 10
E = 1
B = 20
is_iid = False
server_opt = "Adam"
client_opt = "Adam"
client_opt_strategy = "nothing"
wrong_lrs = []
server_lr = 10
client_lr = 0.0000001
# if any(
#     [
#         (wslr <= server_lr) and (wclr <= client_lr)
#         for wslr, wclr in wrong_lrs
#     ]
# ):
#     continue

# TODO a paraméterek helytelen nevére nem adott hibát
config = TorchFederatedLearnerCIFAR100Config(
    BREAK_ROUND=300,
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=client_opt,
    CLIENT_OPT_ARGS=get_args(client_opt),
    # CLIENT_OPT_L2=1e-4,
    CLIENT_OPT_STRATEGY=client_opt_strategy,
    SERVER_OPT=server_opt,
    SERVER_OPT_ARGS=get_args(server_opt),
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
    INIT="keras",
    STORE_OPT_ON_DISK=False,
)
try:
    do_training(config)
except ToLargeLearningRateExcpetion:
    wrong_lrs.append((server_lr, client_lr))
