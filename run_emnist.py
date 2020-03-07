from comet_ml import Experiment
import tensorflow as tf
from FederatedEMNISTLearner import FederatedMNISTLearner
from FederatedLearner import FederatedLearnerConfig

experiment = Experiment(workspace="federated-learning", project_name="emnist")
experiment.set_name("00_tff_tutorial")
config = FederatedLearnerConfig()
config.N_ROUNDS = 200
learner = FederatedMNISTLearner(experiment, config)
learner.train()
