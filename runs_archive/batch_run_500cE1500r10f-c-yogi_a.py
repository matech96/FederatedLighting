from comet_ml import Experiment
from FLF.TorchFederatedLearnerCIFAR100 import TorchFederatedLearnerCIFAR100Config
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig

import common

server_lr = 1
client_lr = 0.001
server_opt = "SGD"
client_opt = "Yogi"
client_opt_strategy = "avg"
max_rounds = 1500
n_clients_per_round = 10
NC = 500
C = n_clients_per_round / NC
# E = 5
B = 20
is_iid = False
# image_norm = "tflike"
# TODO a paraméterek helytelen nevére nem adott hibát
param_names = [
    "N_EPOCH_PER_CLIENT",
]
config_changes = [5, 10]

for values in config_changes:
    project_name = (
        f"{NC}c{values}e{max_rounds}r{n_clients_per_round}f-{server_opt}-compare"
    )
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
        # N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=max_rounds,
        IMAGE_NORM="recordwisefull",
        NORM="group",
        INIT="tffed",
        AUG="basicf",
    )
    if len(param_names) == 1:
        setattr(config, param_names[0], values)
    else:
        for k, v in zip(param_names, values):
            setattr(config, k, v)
    config_technical = TorchFederatedLearnerTechnicalConfig()
    name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    experiment = Experiment(
        workspace="federated-learning-hpopt", project_name=project_name
    )
    common.do_training(experiment, name, config, config_technical)
