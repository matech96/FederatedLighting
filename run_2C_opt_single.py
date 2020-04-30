from comet_ml import Experiment
import sys
import multiprocessing
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


C = 1
NC = 100
E = 3
B = 64
is_iid = False
opt = "SGD"
opt_strategy = "nothing"
lr = 0.1
configs = []


config = TorchFederatedLearnerCIFAR100Config(
    LEARNING_RATE=1,
    OPT=opt,
    OPT_STRATEGY=opt_strategy,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=5,
    MAX_ROUNDS=2,
    DL_N_WORKER=0,
)
configs.append(config)

for lr in [0.001, 0.01, 0.1]:
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = TorchFederatedLearnerCIFAR100Config(
        LEARNING_RATE=lr,
        OPT=opt,
        OPT_STRATEGY=opt_strategy,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=2,
        DL_N_WORKER=0,
    )
    configs.append(config)


def do_training(config: TorchFederatedLearnerCIFAR100Config):
    name = f"{config.OPT} - {config.OPT_STRATEGY} - {config.LEARNING_RATE} - {config.N_EPOCH_PER_CLIENT}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name="cifar")
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config)
    learner.train()


pool = multiprocessing.Pool(2)
pool.map(do_training, configs)
