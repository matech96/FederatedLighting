import torch as th


def get_avg_results(ptr):
    n_childs = len(ptr.child.child)
    return ptr.get(sum_results=True) / n_childs


def avg_models(models):
    final_model = models[0].copy()

    parameters = [model.state_dict() for model in models]
    final_state_dict = avg_model_state_dicts(parameters)
    final_model.load_state_dict(final_state_dict)

    return final_model


# def lambda_model_state_dicts(state_dict, fnc):
#     with th.no_grad():
#         for parameter_name in state_dict.keys():
#             state_dict[parameter_name] = fnc(state_dict[parameter_name])
#     return state_dict


# def lambda2_model_state_dicts(state_dict_0, state_dict_1, fnc):
#     final_state_dict = {}
#     with th.no_grad():
#         for parameter_name in state_dict_0.keys():
#             final_state_dict[parameter_name] = fnc(
#                 state_dict_0[parameter_name], state_dict_1[parameter_name]
#             )
#     return final_state_dict


def lambda_params(params, fnc):
    with th.no_grad():
        return [fnc(p) for p in params]


def lambda2_params(params_0, params_1, fnc):
    with th.no_grad():
        return [fnc(p0, p1) for p0, p1 in zip(params_0, params_1)]


def avg_model_state_dicts(state_dicts):
    final_state_dict = {}
    with th.no_grad():
        for parameter_name in state_dicts[0].keys():
            if (not isinstance(state_dicts[0][parameter_name], th.Tensor)) or (
                state_dicts[0][parameter_name].dtype == th.int64
            ):
                final_state_dict[parameter_name] = state_dicts[0][parameter_name]
                # TODO assert equivalnce
                continue

            final_state_dict[parameter_name] = th.mean(
                th.stack(
                    [
                        model_parameters[parameter_name]
                        for model_parameters in state_dicts
                    ]
                ),
                dim=0,
            )
    return final_state_dict


def commulative_avg_models(model_0, model_1, n_models_0):
    final_model = model_0.copy()

    final_state_dict = commulative_avg_model_state_dicts(
        model_0.state_dict(), model_1.state_dict(), n_models_0
    )
    final_model.load_state_dict(final_state_dict)

    return final_model


def commulative_avg_model_state_dicts(state_dict_0, state_dict_1, n_states_0):
    if state_dict_0 is None:
        return state_dict_1
    final_state_dict = {}
    with th.no_grad():
        for parameter_name in state_dict_0.keys():
            final_state_dict[parameter_name] = (
                (state_dict_0[parameter_name] * n_states_0)
                + state_dict_1[parameter_name]
            ) / (n_states_0 + 1)
    return final_state_dict


def commulative_avg_params(params_0, params_1, n_states_0):
    if params_0 is None:
        return params_1

    return lambda2_params(
        params_0, params_1, lambda a, b: (a * n_states_0 + b) / (n_states_0 + 1)
    )
