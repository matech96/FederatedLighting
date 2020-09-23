from comet_ml import Experiment
import common

from FLF.TorchFederatedLearnerCIFAR100 import TorchFederatedLearnerCIFAR100Config
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.model.torchinit import TorchInitRepo

project_name = "server-yogi-l2"

max_rounds = 1500
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

for l2 in [1e-3, 1e-2, 1e-1, 1e-1, 1e1]:
    s_opt_args = common.get_args(server_opt)
    s_opt_args["weight_decay"] = l2
    config = TorchFederatedLearnerCIFAR100Config(
        BREAK_ROUND=300,
        CLIENT_LEARNING_RATE=client_lr,
        CLIENT_OPT=client_opt,
        CLIENT_OPT_ARGS=common.get_args(client_opt),
        CLIENT_OPT_L2=l2,
        CLIENT_OPT_STRATEGY=client_opt_strategy,
        SERVER_OPT=server_opt,
        SERVER_OPT_ARGS=s_opt_args,
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
        INIT="tffed",
    )
    config_technical = TorchFederatedLearnerTechnicalConfig(HIST_SAMPLE=0)
    name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    common.do_training(experiment, name, config, config_technical)
