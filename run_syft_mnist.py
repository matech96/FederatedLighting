from comet_ml import Experiment
import logging
from SyftFederatedLearner import SyftFederatedLearnerConfig
from SyftFederatedLearnerMNIST import SyftFederatedLearnerMNIST

logging.basicConfig(level=logging.INFO)

for n_epochs in [1, 2, 3, 4, 5]:
    name = f"Number of epochs: {n_epochs}"
    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name="increased_computation")
    experiment.set_name(name)
    config = SyftFederatedLearnerConfig(N_CLIENTS=5, N_EPOCH_PER_CLIENT=n_epochs)
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()

# TODO n workers for data loading is broken
