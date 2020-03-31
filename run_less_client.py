from comet_ml import Experiment
import multiprocessing as mp
from SyftFederatedLearnerMNIST import (
    SyftFederatedLearnerMNIST,
    SyftFederatedLearnerMNISTConfig,
)


def train(experiment, config):
    learner = SyftFederatedLearnerMNIST(experiment, config)
    learner.train()


C = 1
E = 5
B = 50
experiments = []
configs = []
for is_iid in [False, True]:
    for NC in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        dist = "IID" if is_iid else "non IID"
        name = f"{dist} - {NC}"

        experiment = Experiment(
            workspace="federated-learning", project_name="less_client"
        )
        experiment.set_name(name)
        # TODO a paraméterek helytelen nevére nem adott hibát
        config = SyftFederatedLearnerMNISTConfig(
            LEARNING_RATE=0.1,
            IS_IID_DATA=is_iid,
            BATCH_SIZE=B,
            CLIENT_FRACTION=C,
            N_CLIENTS=NC,
            N_EPOCH_PER_CLIENT=E,
            MAX_ROUNDS=1500,
        )
        experiments.append(experiment)
        configs.append(config)

with mp.Pool(2) as p:
    p.map(train, zip(experiments, configs))
