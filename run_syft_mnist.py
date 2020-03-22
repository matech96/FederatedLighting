from comet_ml import Experiment
import logging
from SyftFederatedLearnerMNIST import SyftFederatedLearnerMNIST, SyftFederatedLearnerMNISTConfig

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

for is_iid in [False, True]:
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
