from comet_ml import Experiment

import logging
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.BreakedTrainingExcpetion import ToLargeLearningRateExcpetion


def do_training(config: TorchFederatedLearnerCIFAR100Config):
    name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config)
    learner.train()


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

project_name = "both-side-opt"

max_rounds = 1500
C = 10 / 500
NC = 500
E = 1
B = 20
is_iid = False
# server_lr = 0.1
server_opt = "SGD"
client_opt = "SGD"
client_opt_strategy = "avg"
# image_norm = "tflike"
wrong_lrs = []
for server_lr in [0.01, 0.1, 1, 10]:
    for client_lr in [0.001, 0.0001]:
        if (server_lr != 10) and (client_lr == 0.001):
            continue

        if any(
            [((wslr <= server_lr) and (wclr <= client_lr)) for wslr, wclr in wrong_lrs]
        ):
            continue

        # TODO a paraméterek helytelen nevére nem adott hibát
        config = TorchFederatedLearnerCIFAR100Config(
            BREAK_ROUND=100,
            CLIENT_LEARNING_RATE=client_lr,
            CLIENT_OPT=client_opt,
            # CLIENT_OPT_L2=1e-4,
            CLIENT_OPT_ARGS={"momentum": 0.9},
            CLIENT_OPT_STRATEGY=client_opt_strategy,
            SERVER_OPT=server_opt,
            # SERVER_OPT_ARGS={"betas": (0.0, 0.99), "eps": 0.01},
            SERVER_OPT_ARGS={"momentum": 0.9},
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
        )
        try:
            do_training(config)
        except ToLargeLearningRateExcpetion:
            wrong_lrs.append((server_lr, client_lr))
