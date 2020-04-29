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
NC = 500
E = 5
B = 64
is_iid = False
lr = 0.01
opt = "SGD"
opt_strategy = "nothing"
name = f"{opt} - {opt_strategy} - {lr} - {E}"

logging.info(name)
experiment = Experiment(workspace="federated-learning", project_name="cifar")
experiment.set_name(name)
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
    MAX_ROUNDS=100,
)
learner = TorchFederatedLearnerCIFAR100(experiment, config)
learner.train()
