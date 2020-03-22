from comet_ml import Experiment
import sys
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

for is_iid in [False, True]:
    for B in [sys.maxsize, 10]:
        for C in [0.0, 0.1, 0.2, 0.5, 1.0]:
            dist = "IID" if is_iid else "non IID"
            name = f"{dist} - {B} - {C}"

            logging.info(name)
            experiment = Experiment(
                workspace="federated-learning", project_name="Increasing parallelism"
            )
            experiment.set_name(name)
            config = SyftFederatedLearnerMNISTConfig(
                IS_IID_DATA=is_iid,
                BATCH_SIZE=B,
                CLIENT_FRACTION=C,
                N_CLIENTS=100,
                N_EPOCH_PER_CLIENT=5,
                N_ROUNDS=1200,
            )
            learner = SyftFederatedLearnerMNIST(experiment, config)
            learner.train()

# TODO n workers for data loading is broken
