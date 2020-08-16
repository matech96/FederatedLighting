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
