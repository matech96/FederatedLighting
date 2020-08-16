import comet_ml # Comet.ml needs to be imported before PyTorch
import torch as th

import logging
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.hyperopt.AdvancedGridLearningRate import explore_lr


server_lr = 0.01
client_lr = 0.0001
server_opt = "Adam"
client_opt = "SGD"
client_opt_strategy = "reinit"
project_name = "10-clients"

max_rounds = 30
C = 1
NC = 10
E = 1
B = 20
is_iid = False

config_technical = TorchFederatedLearnerTechnicalConfig(
    STORE_OPT_ON_DISK=False, STORE_MODEL_IN_RAM=False, DL_N_WORKER=0,
)


def get_args(opt):
    if opt == "Adam":
        return {"betas": (0.0, 0.99), "eps": 0.01}
    elif opt == "SGD":
        return {"momentum": 0.9}
    else:
        return {}


th.cuda.set_device(3)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

config = TorchFederatedLearnerCIFAR100Config(
    BREAK_ROUND=300,
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=client_opt,
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
    NORM="group",
    INIT="keras",
)

explore_lr(project_name, TorchFederatedLearnerCIFAR100, config, config_technical)
