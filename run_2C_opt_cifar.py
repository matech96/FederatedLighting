from comet_ml import Experiment
import logging
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)

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
for opt_strategy in ["nothing", "reinit", "avg"]:
    for lr in [0.001, 0.1, 0.0001]:
        for opt in ["SGD", "ASGD", "Adadelta", "Adam"]:
            name = f"{opt} - {opt_strategy} - {lr} - {E}"

            logging.info(name)
            experiment = Experiment(
                workspace="federated-learning", project_name="2C_opt_cifar"
            )
            experiment.set_name(name)
            # TODO a paraméterek helytelen nevére nem adott hibát
            config = TorchFederatedLearnerCIFAR100Config(
                LEARNING_RATE=lr,
                OPT=opt,
                OPT_STRATEGY=opt_strategy,
                IS_IID_DATA=is_iid,
                BATCH_SIZE=B,
                CLIENT_FRACTION=C,
                N_CLIENTS=NC,
                N_EPOCH_PER_CLIENT=E,
                MAX_ROUNDS=100,
                DL_N_WORKER=4,
            )
            learner = TorchFederatedLearnerCIFAR100(experiment, config)
            learner.train()
