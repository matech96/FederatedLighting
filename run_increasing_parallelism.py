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

B = 600  # 10

for is_iid in [True, False]:
    for C in [0.0, 0.1, 0.2, 0.5, 1.0]:
        dist = "IID" if is_iid else "non IID"
        name = f"{dist} - {B} - {C}"

        logging.info(name)
        experiment = Experiment(
            workspace="federated-learning", project_name="Increasing parallelism"
        )
        experiment.set_name(name)
        # TODO a paraméterek helytelen nevére nem adott hibát
        config = SyftFederatedLearnerMNISTConfig(
            LEARNING_RATE=0.1,
            IS_IID_DATA=is_iid,
            BATCH_SIZE=B,
            CLIENT_FRACTION=C,
            N_CLIENTS=100,
            N_EPOCH_PER_CLIENT=5,
            MAX_ROUNDS=1500,
        )
        learner = SyftFederatedLearnerMNIST(experiment, config)
        learner.train()
