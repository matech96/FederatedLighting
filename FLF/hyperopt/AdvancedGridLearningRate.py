import functools
import operator
import logging
import pandas as pd
import comet_ml
from comet_ml import Experiment
from comet_ml.query import Parameter

from typing import Callable

from FLF.TorchFederatedLearner import (
    TorchFederatedLearner,
    TorchFederatedLearnerConfig,
    TorchFederatedLearnerTechnicalConfig,
)
from FLF.BreakedTrainingExcpetion import ToLargeLearningRateExcpetion
from mutil.cometml.apiquery import exp_metrics2list, exp_params2list


def explore_lr(
    project_name: str,
    Learner: Callable[
        [
            Experiment,
            TorchFederatedLearnerConfig,
            TorchFederatedLearnerTechnicalConfig,
        ],
        TorchFederatedLearner,
    ],
    config: TorchFederatedLearnerConfig,
    config_technical: TorchFederatedLearnerTechnicalConfig,
    workspace: str = "federated-learning-hpopt",
    is_continue: bool = False,
):
    db = AdvancedGridLearningRate(
        project_name, Learner, config, config_technical, workspace, is_continue
    )
    for c_lr, s_lr in db:
        db.train(c_lr, s_lr)


class AdvancedGridLearningRate:
    def __init__(
        self,
        project_name: str,
        Learner: Callable[
            [
                Experiment,
                TorchFederatedLearnerConfig,
                TorchFederatedLearnerTechnicalConfig,
            ],
            TorchFederatedLearner,
        ],
        config: TorchFederatedLearnerConfig,
        config_technical: TorchFederatedLearnerTechnicalConfig,
        workspace: str,
        is_continue: bool = False,
    ):
        self.comet_api = comet_ml.api.API()
        self.df = pd.DataFrame(columns=["c_lr", "s_lr", "acc"])
        self.workspace = workspace
        self.project_name = project_name
        self.Learner = Learner
        self.config = config
        self.config_technical = config_technical

        if not is_continue:
            self.__train_config()
        else:
            self.__refresh_df()

    def __train_config(self):
        name = f"{self.config.SERVER_OPT}: {self.config.SERVER_LEARNING_RATE} - {self.config.CLIENT_OPT_STRATEGY} - {self.config.CLIENT_OPT}: {self.config.CLIENT_LEARNING_RATE}"
        logging.info(name)
        experiment = Experiment(
            workspace=self.workspace, project_name=self.project_name
        )
        experiment.set_name(name)
        learner = self.Learner(experiment, self.config, self.config_technical)
        try:
            learner.train()
        except ToLargeLearningRateExcpetion:
            pass  # TODO

        self.__refresh_df()

    def __refresh_df(self):
        parameter_list = [
            Parameter(k) == v if not isinstance(v, bool) else Parameter(k) == str(v)
            for k, v in self.config.flatten().items()
            if not k.endswith("LEARNING_RATE")
        ]
        query = functools.reduce(operator.and_, parameter_list,)
        exps = self.comet_api.query(self.workspace, self.project_name, query)
        self.df = _get_df(exps)

    def __iter__(self):
        return self

    def __next__(self):
        max_series = self.df.iloc[self.df.acc.idxmax()]
        max_c_lr = max_series.c_lr
        max_s_lr = max_series.s_lr
        if (self.config.SERVER_OPT == "SGD") and ("momentum" not in self.config.SERER_OPT_ARGS.keys()):
            mult = [1 / 10, 10]
            for c_lr_m in mult:
                c_lr = max_c_lr * c_lr_m
                if self.get(c_lr, max_s_lr) == -1:
                    return c_lr, max_s_lr
        else:
            mult = [1 / 10, 1, 10]
            for s_lr_m in mult:
                for c_lr_m in mult:
                    c_lr = max_c_lr * c_lr_m
                    s_lr = max_s_lr * s_lr_m
                    if self.get(c_lr, s_lr) == -1:
                        return c_lr, s_lr
        raise StopIteration

    def train(self, c_lr, s_lr):
        self.config.CLIENT_LEARNING_RATE = c_lr
        self.config.SERVER_LEARNING_RATE = s_lr
        self.__train_config()

    def get(self, c_lr, s_lr):
        res = self.df[
            (abs(self.df.c_lr - c_lr) < 1e-15) & (abs(self.df.s_lr - s_lr) < 1e-15)
        ]["acc"]
        if len(res) == 0:
            return -1
        elif len(res) == 1:
            return res.iloc[0]
        else:
            print(len(res))
            print(
                self.df[
                    (abs(self.df.c_lr - c_lr) < 1e-15)
                    & (abs(self.df.s_lr - s_lr) < 1e-15)
                ]
            )


def _get_df(exps):
    c_lr = exp_params2list(exps, "CLIENT_LEARNING_RATE", float)
    s_lr = exp_params2list(exps, "SERVER_LEARNING_RATE", float)
    teas = exp_metrics2list(exps, "test_acc", float)
    return pd.DataFrame({"acc": teas, "c_lr": c_lr, "s_lr": s_lr})
