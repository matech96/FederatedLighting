from comet_ml import Experiment
import tensorflow as tf
from FederatedEMNISTLearner import FederatedEMNISTLearner, FederatedEMNISTLearnerConfig

experiment = Experiment(workspace="federated-learning", project_name="emnist")
experiment.set_name("00_tff_tutorial")
config = FederatedEMNISTLearnerConfig(N_ROUNDS=5, TEST_AFTER=1)
learner = FederatedEMNISTLearner(experiment, config)
learner.train()
