from comet_ml import Experiment
import logging
from SyftFederatedLearner import SyftFederatedLearnerConfig
from SyftFederatedLearnerMNIST import SyftFederatedLearnerMNIST

logging.basicConfig(level=logging.INFO)

for n_clients in [1, 2, 3, 4, 5]:
    name = f"Number of clients: {n_clients}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name="syft mnist")
    experiment.set_name(name)
    config = SyftFederatedLearnerConfig(N_CLIENTS=n_clients)
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()
