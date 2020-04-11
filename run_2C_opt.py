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
is_iid = False
for opt in TorchOptRepo.get_opt_names()[1:]:
    for lr in [0.1, 0.01]:
        name = f"{opt} - {lr} - {E}"

        logging.info(name)
        experiment = Experiment(workspace="federated-learning", project_name="2C_opt")
        experiment.set_name(name)
        experiment.log_parameter("opt_srategy", "reinit")
        # TODO a paraméterek helytelen nevére nem adott hibát
        config = TorchFederatedLearnerMNISTConfig(
            LEARNING_RATE=lr,
            OPT=opt,
            IS_IID_DATA=is_iid,
            BATCH_SIZE=B,
            CLIENT_FRACTION=C,
            N_CLIENTS=NC,
            N_EPOCH_PER_CLIENT=E,
            MAX_ROUNDS=100,
        )
        learner = TorchFederatedLearnerMNIST(experiment, config)
        learner.train()
