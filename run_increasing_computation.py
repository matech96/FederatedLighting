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

C = 0.1

for is_iid in [False]:
    for E in [1, 5, 20]:
        for B in [10, 50, 600]:
            dist = "IID" if is_iid else "non IID"
            name = f"{dist} - {E} - {B}"

            logging.info(name)
            experiment = Experiment(
                workspace="federated-learning", project_name="Increasing computation"
            )
            experiment.set_name(name)
            # TODO a paraméterek helytelen nevére nem adott hibát
            config = SyftFederatedLearnerMNISTConfig(
                LEARNING_RATE=0.1,
                IS_IID_DATA=is_iid,
                BATCH_SIZE=B,
                CLIENT_FRACTION=C,
                N_CLIENTS=100,
                N_EPOCH_PER_CLIENT=E,
                MAX_ROUNDS=1500,
            )
            learner = SyftFederatedLearnerMNIST(experiment, config)
            learner.train()
