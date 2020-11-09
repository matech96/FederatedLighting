import pandas as pd
import numpy as np
from collections import defaultdict

from comet_ml import APIExperiment
from typing import List, Callable, Any


def exp_params2list(exps, n, t):
    return __exps2list(exps, lambda exp: exp.get_parameters_summary(n), t)


def exp_metrics2list(exps, n, t):
    return __exps2list(exps, lambda exp: exp.get_metrics_summary(n), t)


def __exps2list(
    exps: List[APIExperiment],
    c: Callable[[APIExperiment], Any],
    t: Callable[[str], Any],
):
    res = []
    for exp in exps:
        response = c(exp)
        if isinstance(response, list):
            res.append(None)
        else:
            res.append(t(response["valueMax"]))
    return res


class ExperminetInfo:
    def __init__(self, comet_api, exp_id):
        self.exp = comet_api.get(exp_id)
        self.__metrics = None

    @property
    def metrics(self):
        if self.__metrics is None:
            self.__metrics = self.exp.get_metrics()
        return self.__metrics

    @property
    def metric_names(self):
        return np.unique([m["metricName"] for m in self.metrics])

    def get_metrics_df(self, metric_names=None):
        if metric_names is None:
            metric_names = self.metric_names

        values_dict = defaultdict(lambda: {"d": [], "i": []})
        for metric_name in metric_names:
            metrics = self.exp.get_metrics(metric_name)
            for m in metrics:
                if m["step"] is not None:
                    v = values_dict[m["metricName"]]
                    v["d"].append(float(m["metricValue"]))
                    v["i"].append(m["step"])
        return pd.DataFrame(
            {k: pd.Series(v["d"], v["i"]) for k, v in values_dict.items()}
        )

    def get_parameter(self, parameter_name, t):
        return t(self.exp.get_parameters_summary(parameter_name)["valueCurrent"])
