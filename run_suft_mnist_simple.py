from comet_ml import Experiment
import logging
import sys

from SyftFederatedLearnerMNIST import SyftFederatedLearnerMNIST, SyftFederatedLearnerMNISTConfig

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

name = "ClientBatchIter"
logging.info(name)
experiment = Experiment(
    workspace="federated-learning", project_name="simple_runs"
)
experiment.set_name(name)
config = SyftFederatedLearnerMNISTConfig(N_CLIENTS=100, CLIENT_FRACTION=0.1, N_ROUNDS=10, BATCH_SIZE=64)
learner = SyftFederatedLearnerMNIST(experiment, config)
learner.train()

# TODO n workers for data loading is broken
