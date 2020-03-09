from comet_ml import Experiment
import logging
import sys
from FederatedEMNISTLearner import FederatedEMNISTLearner, FederatedEMNISTLearnerConfig

logging.basicConfig(level=logging.INFO)
sys.setrecursionlimit(6000)
logging.info(f"Recursion limit is {sys.getrecursionlimit()}.")

for n_clients in range(1, 11):
    logging.info(f"Running with {n_clients}")
    experiment = Experiment(workspace="federated-learning", project_name="emnist")
    experiment.set_name(f"Client test: {n_clients}")
    config = FederatedEMNISTLearnerConfig(
        N_ROUNDS=200, TEST_AFTER=10, N_CLIENTS=n_clients
    )
    learner = FederatedEMNISTLearner(experiment, config)
    learner.train()
