import numpy as np
import pandas as pd
import re
import holoviews as hv
from bokeh.io import export_svgs
from comet_ml.exceptions import NotFound

from mutil.cometml.apiquery import exp_metrics2list, exp_params2list
from mutil.hv import SublabelProvider


def get_label(sopt, strat, copt, scf):
    if strat.lower() == "a":
        notation = f"{sopt}-(A)-{copt}"
    else:
        notation = f"{sopt}-{copt}"

    if scf:
        notation += " + SCAFFOLD"
    return notation


def generate_single_plot(comet_api, SOPT, STR="r", COPT="sgd", E=1, project="emnist-s"):
    clr = "Client Learning Rate (log10)"
    slr = "Server Learning Rate (log10)"
    exps = get_experiments(comet_api, SOPT, STR, COPT, E, project)
    slrv = exp_params2list(exps, "SERVER_LEARNING_RATE", float)
    clrv = exp_params2list(exps, "CLIENT_LEARNING_RATE", float)
    acc = exp_metrics2list(exps, "last_avg_acc", float)
    df = pd.DataFrame({slr: np.log10(slrv), clr: np.log10(clrv), "Accuracy": acc})
    i = df["Accuracy"].idxmax()
    m = df.iloc[i]
    return (
        hv.HeatMap(df, kdims=[clr, slr]).opts(colorbar=True, cmap="viridis")
        * hv.Ellipse(m[clr], m[slr], 0.5)
    ).opts(fig_inches=10)


def get_plot_dict(comet_api, SOPT, STR="r", COPT="sgd", project="emnist-s"):
    plots = {}
    for E in [1, 5, 10, 20, 30]:
        try:
            plots[E] = generate_single_plot(comet_api, SOPT, STR, COPT, E, project)
        except NotFound:
            break
    return plots


def save_plot(comet_api, SOPT, STR="R", COPT="SGD", project="emnist-s"):
    plots = get_plot_dict(comet_api, SOPT, STR, COPT, project)
    plot = None
    for E, p in plots.items():
        if plot is None:
            plot = p
        else:
            plot += p
    p = plot.opts(
        fig_size=200,
        title=get_label(SOPT, STR, COPT, False),
        tight=True,
        sublabel_format=SublabelProvider(plots.keys()),
    ).cols(2)
    hv.save(p, f"lr_{SOPT}_{STR}_{COPT}.svg")


def generate_plot(comet_api, SOPT, STR="R", COPT="SGD", project="emnist-s"):
    plots = get_plot_dict(comet_api, SOPT, STR, COPT, project)
    return hv.HoloMap(plots, kdims="E").opts(title=get_label(SOPT, STR, COPT, False))


def get_experiments(comet_api, SOPT, STR="r", COPT="sgd", E=1, project="emnist-s"):
    workspace = f"federated-learning-{project}"
    projs = comet_api.get(workspace)
    r = re.compile(
        r"^cnn(?P<NC>\d+)c(?P<E>\d+)e(?P<max_rounds>\d+)r(?P<n_clients_per_round>\d+)f-(?P<server_opt>\w+)-(?P<client_opt_strategy>\w+)-(?P<client_opt>\w+)$"
    )
    for proj in projs:
        m = r.search(proj)
        if m is None:
            continue
        if (
            (int(m.group("E")) == E)
            and (m.group("server_opt") == SOPT.lower())
            and (m.group("client_opt") == COPT.lower())
            and (m.group("client_opt_strategy") == STR.lower())
        ):
            q = f"{workspace}/{proj}"
            return comet_api.get(q)


def save_bokeh_svg(x, fname="p.svg"):
    p = hv.render(x, backend='bokeh')
    p.output_backend = "svg"
    export_svgs(p, filename=fname)
