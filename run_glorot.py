from comet_ml import Experiment
import logging
from TorchFederatedLearnerMNIST import (
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
E = 5
B = 50
is_iid = False

dist = "IID" if is_iid else "non IID"
name = f"torch - {dist} - glorot"

logging.info(name)
experiment = Experiment(
    workspace="federated-learning", project_name="compare-frameworks"
)
experiment.set_name(name)
# TODO a paraméterek helytelen nevére nem adott hibát
config = TorchFederatedLearnerMNISTConfig(
    LEARNING_RATE=0.1,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=1500,
)
learner = TorchFederatedLearnerMNIST(experiment, config)
learner.train()
