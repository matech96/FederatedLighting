# Trains SCAFFOLD on the EMNIST dataset

from FLF.TorchFederatedLearner import (
    TorchFederatedLearnerTechnicalConfig,
)
from FLF.TorchFederatedLearnerEMNIST import TorchFederatedLearnerEMNISTConfig

import common
from mutil.Empty import Empty


server_lr = 1
client_lr = 0.31
server_opt = "SGD"
client_opt = "SGD"
client_opt_strategy = "reinit"

max_rounds = 10
n_clients_per_round = 10
NC = 3400
C = n_clients_per_round / NC
B = 20
is_iid = False
model = "CNN"
E = 1

project_name = f"{model}{NC}c{E}e{max_rounds}r{n_clients_per_round}f-{server_opt}-{client_opt_strategy[0]}-{client_opt}-scf"


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
    MODEL=model,
    SCAFFOLD=True
)
config_technical = TorchFederatedLearnerTechnicalConfig(
    BREAK_ROUND=300,
    EVAL_ROUND=1,
    TEST_LAST=1,
    STORE_OPT_ON_DISK=False,
    STORE_MODEL_IN_RAM=False,
)
name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
experiment = Empty()
common.do_training_emnist(experiment, name, config, config_technical)
