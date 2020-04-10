from comet_ml import Experiment
import logging

from FFL.TorchFederatedLearnerMNIST import TorchFederatedLearnerMNIST, TorchFederatedLearnerMNISTConfig

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

name = "TorchClientBatchIter"
logging.info(name)
experiment = Experiment(
    workspace="federated-learning", project_name="simple_runs"
)
experiment.set_name(name)
config = TorchFederatedLearnerMNISTConfig(N_CLIENTS=100, CLIENT_FRACTION=0.1, N_ROUNDS=10, BATCH_SIZE=64)
learner = TorchFederatedLearnerMNIST(experiment, config)
learner.train()

# TODO n workers for data loading is broken
