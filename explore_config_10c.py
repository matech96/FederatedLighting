import comet_ml  # Comet.ml needs to be imported before PyTorch

from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.hyperopt.AdvancedGridLearningRate import explore_lr
import common


server_lr = 0.1
client_lr = 0.0001
server_opt = "Yogi"
client_opt = "Yogi"
client_opt_strategy = "avg"

max_rounds = 30  # 1500
C = 1  # 10 / 500
NC = 10  # 500
E = 1
B = 20
is_iid = False
project_name = f"{NC}c{E}e-{server_opt}-{client_opt_strategy}-{client_opt}"

config_technical = TorchFederatedLearnerTechnicalConfig(
    STORE_OPT_ON_DISK=False, STORE_MODEL_IN_RAM=False, BREAK_ROUND=5
)

config = TorchFederatedLearnerCIFAR100Config(
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=client_opt,
    CLIENT_OPT_ARGS=common.get_args(client_opt),
    CLIENT_OPT_L2=1e-4,
    CLIENT_OPT_STRATEGY=client_opt_strategy,
    SERVER_OPT=server_opt,
    SERVER_OPT_ARGS=common.get_args(server_opt),
    SERVER_LEARNING_RATE=server_lr,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=max_rounds,
    IMAGE_NORM="recordwisefull",
    NORM="group",
    INIT="tffed",
    AUG="flipf",
)

explore_lr(project_name, TorchFederatedLearnerCIFAR100, config, config_technical)
