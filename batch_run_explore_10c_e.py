import argparse

from comet_ml import Experiment
from FLF.TorchFederatedLearnerCIFAR100 import TorchFederatedLearnerCIFAR100, TorchFederatedLearnerCIFAR100Config
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.hyperopt.AdvancedGridLearningRate import explore_lr

import common

parser = argparse.ArgumentParser()
parser.add_argument("E", type=int)
args = parser.parse_args()

max_rounds = 30
C = 1
NC = 10
E = args.E
B = 20
is_iid = False
# image_norm = "tflike"
# TODO a paraméterek helytelen nevére nem adott hibát
param_names = [
    "SERVER_OPT",
    "CLIENT_OPT",
    "SERVER_LEARNING_RATE",
    "CLIENT_LEARNING_RATE",
    "CLIENT_OPT_STRATEGY",
]
config_changes = [
    ("SGD", "SGD", 1, 0.1, "nothing"),
    ("Yogi", "SGD", 0.1, 0.01, "nothing"),
    ("Yogi", "Yogi", 0.1, 0.0001, "avg"),
    ("Yogi", "Yogi", 0.1, 0.0001, "reinit"),
    ("Yogi", "Yogi", 0.1, 0.0001, "nothing"),
]
for values in config_changes:
    project_name = f"{NC}c{E}e-{values[0]}-{values[4]}-{values[1]}"
    config = TorchFederatedLearnerCIFAR100Config(
        BREAK_ROUND=1500,
        CLIENT_OPT_L2=1e-4,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=max_rounds,
        IMAGE_NORM="recordwisefull",
        NORM="group",
        INIT="tffed",
        AUG="flipf",
    )
    for k, v in zip(param_names, values):
        setattr(config, k, v)
    config_technical = TorchFederatedLearnerTechnicalConfig(SAVE_CHP_INTERVALL=5, STORE_OPT_ON_DISK=False, STORE_MODEL_IN_RAM=False)
    explore_lr(project_name, TorchFederatedLearnerCIFAR100, config, config_technical)
