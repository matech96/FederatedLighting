import comet_ml  # Comet.ml needs to be imported before PyTorch
import torch as th

from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.TorchFederatedLearnerEMNIST import (
    TorchFederatedLearnerEMNIST,
    TorchFederatedLearnerEMNISTConfig,
)
from FLF.hyperopt.AdvancedGridLearningRate import explore_lr
import common


server_lr = 1.0
client_lr = 0.1
server_opt = "SGD"
client_opt = "SGD"
client_opt_strategy = "reinit"

max_rounds = 1500
n_clients_per_round = 10
NC = 3400
C = n_clients_per_round / NC
E = 50
B = 20
is_iid = False
project_name = f"{NC}c{E}e{max_rounds}r{n_clients_per_round}f-{server_opt}-{client_opt_strategy[0]}-{client_opt}"

config_technical = TorchFederatedLearnerTechnicalConfig(BREAK_ROUND=300)

config = TorchFederatedLearnerEMNISTConfig(
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=common.get_name(client_opt),
    CLIENT_OPT_ARGS=common.get_args(client_opt),
    # CLIENT_OPT_L2=1e-4,
    CLIENT_OPT_STRATEGY=client_opt_strategy,
    SERVER_OPT=common.get_name(server_opt),
    SERVER_OPT_ARGS=common.get_args(server_opt),
    SERVER_LEARNING_RATE=server_lr,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=max_rounds,
)

explore_lr(
    project_name,
    TorchFederatedLearnerEMNIST,
    config,
    config_technical,
    "federated-learning-emnist",
)
