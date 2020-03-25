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

B = 10  # 600
is_iid = True
C = 0.1
lr = 0.1
for i in range(3):
    name = f"no_torch_{i}"

    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="Reproducability seed"
    )
    experiment.set_name(name)
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = SyftFederatedLearnerMNISTConfig(
        LEARNING_RATE=lr,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=100,
        N_EPOCH_PER_CLIENT=5,
        MAX_ROUNDS=300,
        SEED=0,
    )
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()
