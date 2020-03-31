from comet_ml import Experiment
import logging
from SyftFederatedLearnerMNIST import (
    SyftFederatedLearnerMNIST,
    SyftFederatedLearnerMNISTConfig,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

C = 1
E = 5
B = 50

is_iid = False
for NC in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
    dist = "IID" if is_iid else "non IID"
    name = f"{dist} - {NC}"

    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name="less_client")
    experiment.set_name(name)
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = SyftFederatedLearnerMNISTConfig(
        LEARNING_RATE=0.1,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=1500,
        DL_N_WORKER=2,
    )
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()
