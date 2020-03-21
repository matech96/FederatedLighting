from comet_ml import Experiment
import logging
from SyftFederatedLearner import SyftFederatedLearnerConfig
from SyftFederatedLearnerMNIST import SyftFederatedLearnerMNIST

logging.basicConfig(level=logging.INFO)

for client_fraction in [0.0, 0.1, 0.2, 0.5, 1.0]:
    name = f"Client fraction: {client_fraction}"
    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="client_fraction"
    )
    experiment.set_name(name)
    config = SyftFederatedLearnerConfig(N_CLIENTS=100, CLIENT_FRACTION=client_fraction)
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()

# TODO n workers for data loading is broken
