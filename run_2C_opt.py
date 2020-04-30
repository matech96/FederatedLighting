from comet_ml import Experiment
import logging
from FLF.TorchFederatedLearnerMNIST import (
    TorchFederatedLearnerMNIST,
    TorchFederatedLearnerMNISTConfig,
)
from FLF.TorchOptRepo import TorchOptRepo

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


C = 1
NC = 2
E = 1
B = 50
opt_strategy = "avg"
opt = "Rprop"
is_iid = False
for lr in [1e-6, 1e-3]: # [0.01, 0.1, 0.0001, 1, 0.00001, 10]:
    # for opt in TorchOptRepo.get_opt_names():
    name = f"{opt} - {opt_strategy} - {lr} - {E}"

    logging.info(name)
    experiment = Experiment(workspace="federated-learning", project_name="2C_opt")
    experiment.set_name(name)
    experiment.log_parameter("opt_srategy", opt_strategy)
    # TODO a paraméterek helytelen nevére nem adott hibát
    config = TorchFederatedLearnerMNISTConfig(
        LEARNING_RATE=lr,
        OPT=opt,
        OPT_STRATEGY=opt_strategy,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=100,
    )
    learner = TorchFederatedLearnerMNIST(experiment, config)
    learner.train()