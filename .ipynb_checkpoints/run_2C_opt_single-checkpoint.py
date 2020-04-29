from comet_ml import Experiment
import logging
from FLF.TorchFederatedLearnerMNIST import (
    TorchFederatedLearnerMNIST,
    TorchFederatedLearnerMNISTConfig,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


C = 1
NC = 2
E = 1
B = 50
is_iid = False
lr = 1
opt = "Adadelta"
opt_strategy = "nothing"
name = f"{opt} - {opt_strategy} - {lr} - {E}"

logging.info(name)
experiment = Experiment(workspace="federated-learning", project_name="2C_opt")
experiment.set_name(name)
# TODO a paraméterek helytelen nevére nem adott hibát
config = TorchFederatedLearnerMNISTConfig(
    LEARNING_RATE=lr,
    OPT=opt,
    OPT_STRATEGY=opt_strategy,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=100,
)
learner = TorchFederatedLearnerMNIST(experiment, config)
learner.train()
