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

B = 10  # 600
is_iid = True
C = 0.1
lr = 0.1
for s in range(20):
    name = f"Seed value: {s}"

    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="Reproducability seed"
    )
    experiment.set_name(name)
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = TorchFederatedLearnerMNISTConfig(
        LEARNING_RATE=lr,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=100,
        N_EPOCH_PER_CLIENT=5,
        MAX_ROUNDS=300,
        SEED=s,
    )
    learner = TorchFederatedLearnerMNIST(experiment, config)
    learner.train()
