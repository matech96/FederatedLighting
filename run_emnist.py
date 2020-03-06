from comet_ml import Experiment

from FederatedEMNISTLearner import FederatedMNISTLearner

experiment = Experiment(workspace="federated-learning", project_name="emnist")
experiment.set_name('00_tff_tutorial')
learner = FederatedMNISTLearner(experiment)
learner.train(-1, -1)
