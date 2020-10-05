import comet_ml # Comet.ml needs to be imported before PyTorch
import torch as th

from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.hyperopt.AdvancedGridLearningRate import explore_lr
import common


server_lr = 0.01
client_lr = 0.01
server_opt = "SGD"
client_opt = "SGD"
client_opt_strategy = "reinit"
project_name = f"10c40e-s-{server_opt}-c-{client_opt}"

max_rounds = 30  # 1500
C = 0.5  # 10 / 500
NC = 10  # 500
E = 40
B = 20
is_iid = False

config_technical = TorchFederatedLearnerTechnicalConfig()

config = TorchFederatedLearnerCIFAR100Config(
    BREAK_ROUND=300,
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=client_opt,
    # CLIENT_OPT_ARGS=common.get_args(client_opt),
    CLIENT_OPT_L2=1e-4,
    CLIENT_OPT_STRATEGY=client_opt_strategy,
    SERVER_OPT=server_opt,
    # SERVER_OPT_ARGS=common.get_args(server_opt),
    SERVER_LEARNING_RATE=server_lr,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=max_rounds,
    NORM="group",
    INIT="tffed",
)

explore_lr(project_name, TorchFederatedLearnerCIFAR100, config, config_technical)
