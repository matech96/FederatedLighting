import torch as th


def get_avg_results(ptr):
    n_childs = len(ptr.child.child)
    return ptr.get(sum_results=True) / n_childs


def avg_models(models):
    final_model = models[0].copy()
    final_state_dict = {}

    parameters = [model.state_dict() for model in models]
    with th.no_grad():
        for parameter_name in parameters[0].keys():
            final_state_dict[parameter_name] = th.mean(
                th.stack(
                    [
                        model_parameters[parameter_name]
                        for model_parameters in parameters
                    ]
                ),
                dim=0,
            )
        final_model.load_state_dict(final_state_dict)
    return final_model
