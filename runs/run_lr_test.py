from comet_ml import Experiment
import logging
from FFL.TorchFederatedLearnerMNIST import (
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

for lr in [0.1, 0.01]:
    name = f"{lr}"

    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="Learning rate"
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
    )
    learner = TorchFederatedLearnerMNIST(experiment, config)
    learner.train()
