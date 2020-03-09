from comet_ml import Experiment
import tensorflow as tf
from FederatedEMNISTLearner import FederatedEMNISTLearner, FederatedEMNISTLearnerConfig

for n_clients in range(1, 11):
    experiment = Experiment(workspace="federated-learning", project_name="emnist")
    experiment.set_name(f"Client test: {n_clients}")
    config = FederatedEMNISTLearnerConfig(
        N_ROUNDS=200, TEST_AFTER=10, N_CLIENTS=n_clients
    )
    learner = FederatedEMNISTLearner(experiment, config)
    learner.train()
