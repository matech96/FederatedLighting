from comet_ml import Experiment
import logging
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

project_name = "reproduce-FedAvg-CIFAR100"

max_rounds = 1500
C = 10 / 500
NC = 500
E = 1
B = 20
is_iid = False
server_lr = 1
server_opt = "SGD"
client_opt = "SGD"
client_opt_strategy = "reinit"
configs = []

for client_lr in [0.0001, 0.001, 0.01, 0.1, 1]:
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = TorchFederatedLearnerCIFAR100Config(
        CLIENT_LEARNING_RATE=client_lr,
        CLIENT_OPT=client_opt,
        CLIENT_OPT_STRATEGY=client_opt_strategy,
        SERVER_OPT=server_opt,
        # SERVER_OPT_ARGS={"betas": (0.0, 0.99), "eps": 0.01},
        SERVER_LEARNING_RATE=server_lr,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=max_rounds,
        DL_N_WORKER=0,
    )
    configs.append(config)


def do_training(config: TorchFederatedLearnerCIFAR100Config):
    name = f"noB1 {config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config)
    learner.train()


for config in configs:
    do_training(config)
