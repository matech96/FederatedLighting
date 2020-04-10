from comet_ml import Experiment
import logging
import sys
from FederatedEMNISTLearner import FederatedEMNISTLearner, FederatedEMNISTLearnerConfig

logging.basicConfig(level=logging.INFO)
sys.setrecursionlimit(6000)
logging.info(f"Recursion limit is {sys.getrecursionlimit()}.")

for n_writers in [100, 1000, 2000, None]:
    logging.info(f"Writers: {n_writers}")
    experiment = Experiment(workspace="federated-learning", project_name="emnist")
    experiment.set_name(f"{n_writers} Writer")
    config = FederatedEMNISTLearnerConfig(
        N_ROUNDS=10, TEST_AFTER=5, N_CLIENTS=1, WRITER_LIMIT=n_writers
    )
    learner = FederatedEMNISTLearner(experiment, config)
    learner.train()
