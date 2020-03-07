from comet_ml import Experiment
import tensorflow as tf
from FederatedEMNISTLearner import FederatedMNISTLearner
from FederatedLearner import FederatedLearnerConfig

experiment = Experiment(workspace="federated-learning", project_name="emnist")
experiment.set_name("00_tff_tutorial")
config = FederatedLearnerConfig(N_ROUNDS=5)
learner = FederatedMNISTLearner(experiment, config)
learner.train()
