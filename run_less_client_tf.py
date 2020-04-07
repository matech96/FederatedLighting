from comet_ml import Experiment
import logging
from FLF.TensorFlowFederatedLearnerMNIST import TensorFlowFederatedLearnerMNIST, TensorFlowFederatedLearnerMNISTConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

C = 1
NC = 2
E = 5
B = 50
is_iid = False

for E in [1, 2, 3, 4, 5, 10, 20]:
    dist = "IID" if is_iid else "non IID"
    name = f"{dist} - {E}"

    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="2_clients_tensorflow"
    )
    experiment.set_name(name)
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = TensorFlowFederatedLearnerMNISTConfig(
        LEARNING_RATE=0.01,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=1500,
    )
    learner = TensorFlowFederatedLearnerMNIST(experiment, config)
    learner.train()
