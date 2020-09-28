from comet_ml import Experiment
from FLF.TorchFederatedLearnerCIFAR100 import TorchFederatedLearnerCIFAR100Config
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig

import common


server_lr = 0.001
client_lr = 0.01
server_opt = "Yogi"
client_opt = "Yogi"
client_opt_strategy = "reinit"

max_rounds = 1500
n_clients_per_round = 10
NC = 500
C = n_clients_per_round / NC
E = 1
B = 20
is_iid = False
project_name = f"{NC}c{E}e{max_rounds}r{n_clients_per_round}f-{server_lr}-compare"
# TODO a paraméterek helytelen nevére nem adott hibát
config = TorchFederatedLearnerCIFAR100Config(
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=common.get_name(client_opt),
    CLIENT_OPT_ARGS=common.get_args(client_opt),
    CLIENT_OPT_L2=1e-4,
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
    IMAGE_NORM="recordwisefull",
    NORM="group",
    INIT="tffed",
    AUG="basicf",
)
config_technical = TorchFederatedLearnerTechnicalConfig(BREAK_ROUND=300)
name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
experiment = Experiment(workspace="federated-learning", project_name=project_name)
common.do_training(experiment, name, config, config_technical)
