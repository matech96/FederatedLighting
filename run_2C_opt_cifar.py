# from comet_ml import Experiment
# import logging
# from FLF.TorchFederatedLearnerCIFAR100 import (
#     TorchFederatedLearnerCIFAR100,
#     TorchFederatedLearnerCIFAR100Config,
# )

# logging.basicConfig(
#     format="%(asctime)s %(levelname)-8s %(message)s",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


# C = 1
# NC = 2
# E = 1
# B = 50
# is_iid = False
# opt_strategy = "avg"
# for opt_strategy in ["avg", "nothing", "reinit"]:
#     for lr in [0.1, 0.01, 0.001]:
#         for opt in ["SGD", "ASGD", "Adadelta", "Adam"]:
#             name = f"{opt} - {opt_strategy} - {lr} - {E}"

#             logging.info(name)
#             experiment = Experiment(
#                 workspace="federated-learning", project_name="2C_opt_cifar_new"
#             )
#             experiment.set_name(name)
#             # TODO a paraméterek helytelen nevére nem adott hibát
#             config = TorchFederatedLearnerCIFAR100Config(
#                 LEARNING_RATE=lr,
#                 OPT=opt,
#                 OPT_STRATEGY=opt_strategy,
#                 IS_IID_DATA=is_iid,
#                 BATCH_SIZE=B,
#                 CLIENT_FRACTION=C,
#                 N_CLIENTS=NC,
#                 N_EPOCH_PER_CLIENT=E,
#                 MAX_ROUNDS=100,
#                 DL_N_WORKER=0,
#             )
#             learner = TorchFederatedLearnerCIFAR100(experiment, config)
#             learner.train()


from comet_ml import Experiment
import logging
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.TorchOptRepo import TorchOptRepo

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def do_training(config: TorchFederatedLearnerCIFAR100Config):
    name = (
        name
    ) = f"{config.OPT} - {config.OPT_STRATEGY} - {config.LEARNING_RATE} - {config.N_EPOCH_PER_CLIENT}"
    logging.info(name)
    experiment = Experiment(
        workspace="federated-learning", project_name="500C_opt_cifar_new_long"
    )
    experiment.set_name(name)
    learner = TorchFederatedLearnerCIFAR100(experiment, config)
    learner.train()


max_round = 1000
C = 10 / 500
NC = 500
E = 1
B = 50
is_iid = False
# opt_strategy = "nothing"
# lr = 10
# opt = "Adadelta"
configs = []
# for lr in [0.1, 0.01, 0.001, 0.0001, 1, 0.00001]:
options = {"Adagrad": {"avg": 0.01, "reinit": 0.001}, 
            "RMSprop": {"avg": 0.001, "reinit": 0.0001}, 
            "AdamW": {"avg": 0.001, "reinit": 0.001}}
for opt, d in options.items():
    for opt_strategy, lr in d.items():
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
            MAX_ROUNDS=max_round,
            DL_N_WORKER=0,
        )
        configs.append(config)


for _ in range(3):
    for config in configs:
        do_training(config)
