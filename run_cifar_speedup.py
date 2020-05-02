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


C = 1
NC = 100
E = 5
B = 64
is_iid = False
opt = "SGD"
opt_strategy = "nothing"
lr = 0.1
configs = []

lr = 0.1
# for lr in [0.001, 0.01, 0.1]:
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
    name = f"GPU - pinMem - tensors - par0 - {config.OPT} - {config.OPT_STRATEGY} - {config.LEARNING_RATE} - {config.N_EPOCH_PER_CLIENT}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name="cifar-speed-up")
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config)
    learner.train()


for config in configs:
    do_training(config)
