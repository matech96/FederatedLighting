from comet_ml import Experiment
import logging
from SyftFederatedLearnerMNIST import SyftFederatedLearnerMNIST, SyftFederatedLearnerMNISTConfig

logging.basicConfig(level=logging.INFO)

for is_iid in [True, False]:
    if is_iid:
        name = "IID"
    else:
        name = "non IID"
    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="data_distribution"
    )
    experiment.set_name(name)
    config = SyftFederatedLearnerMNISTConfig(N_CLIENTS=100, CLIENT_FRACTION=0.1, N_ROUNDS=10, IS_IID_DATA=is_iid)
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()

# TODO n workers for data loading is broken
