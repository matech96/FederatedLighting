import comet_ml
import common

from FLF.TorchFederatedLearnerCIFAR100 import TorchFederatedLearnerCIFAR100Config
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.model.torchinit import TorchInitRepo

project_name = "yogi_init"

max_rounds = 200
C = 10 / 500
NC = 500
E = 1
B = 20
is_iid = False
server_lr = 0.0316
client_lr = 0.0316
server_opt = "Yogi"
client_opt = "SGD"
client_opt_strategy = "reinit"
# image_norm = "tflike"
# TODO a paraméterek helytelen nevére nem adott hibát

for init in TorchInitRepo.get_opt_names():
    config = TorchFederatedLearnerCIFAR100Config(
        BREAK_ROUND=300,
        CLIENT_LEARNING_RATE=client_lr,
        CLIENT_OPT=client_opt,
        CLIENT_OPT_ARGS=common.get_args(client_opt),
        # CLIENT_OPT_L2=1e-4,
        CLIENT_OPT_STRATEGY=client_opt_strategy,
        SERVER_OPT=server_opt,
        # SERVER_OPT_ARGS=get_args(server_opt),
        SERVER_LEARNING_RATE=server_lr,
        IS_IID_DATA=is_iid,
        BATCH_SIZE=B,
        CLIENT_FRACTION=C,
        N_CLIENTS=NC,
        N_EPOCH_PER_CLIENT=E,
        MAX_ROUNDS=max_rounds,
        DL_N_WORKER=0,
        NORM="group",
        # IMAGE_NORM=image_norm,
        INIT=init,
    )
    config_technical = TorchFederatedLearnerTechnicalConfig(HIST_SAMPLE=0)
    name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    common.do_training(name, project_name, config, config_technical)
