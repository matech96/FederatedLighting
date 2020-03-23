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


def avg_model_state_dicts(state_dicts):
    final_state_dict = {}
    with th.no_grad():
        for parameter_name in state_dicts[0].keys():
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

