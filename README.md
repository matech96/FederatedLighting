# FederatedLighting

## Installation

You only need [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the code.

```bash
docker run -p 8080:8080 --gpus all --env INCLUDE_TUTORIALS=false -v <repository directory>:/workspace -w /workspace matech96/ml-workspace python run_emnist100m_sgd_scf.py
```

## Custom model and data

To use federated learning on your own data, you have to implement the abstract functions of the `TorchFederatedLearner` class:
* [`load_data`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearner.py#L192-L200),
* [`get_model_cls`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearner.py#L203-L209),
* [`get_loss`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearner.py#L212-L218).

Check out the documentation in the code for more detail. Check out the [`TorchFederatedLearnerEMNIST`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearnerEMNIST.py) and [`TorchFederatedLearnerCIFAR100`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearnerCIFAR100.py) classes for concrete examples.

Now you just need to run the training.


```python
from FLF.TorchFederatedLearner import TorchFederatedLearnerConfig, TorchFederatedLearnerTechnicalConfig
from mutil.Empty import Empty

from your_code import TorchFederatedLearnerImplemented

logger = Empty()
config = TorchFederatedLearnerConfig()
config_technical = TorchFederatedLearnerTechnicalConfig()
learner = TorchFederatedLearnerImplemented(logger, config, config_technical)
learner.train()
```

The `logger` variable is meant to be used with a comet.ml experiment object for logging. If you don't use comet.ml `Empty` is a shell class, that does nothing, but doesn't produce any errors.

[`TorchFederatedLearnerConfig`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearner.py#L50-L93) contains parameters that have an effect on the metrics of the training, the parameters of [`TorchFederatedLearnerTechnicalConfig`](https://github.com/matech96/FederatedLighting/blob/master/FLF/TorchFederatedLearner.py#L96-L110) are for technical details (such as the `pin_memory` setting). Both can be extended as presented in the examples. The code contains some explanation for all parameters. 

[comment]: <> (You probably want to review all the paramters in `TorchFederatedLearnerConfig` except for)

[comment]: <> (* `SERVER_LEARNING_RATE`)

[comment]: <> (* `SEED`)

[comment]: <> (* )

The most important parameters are:
* `MAX_ROUNDS`
* `N_CLIENTS`
* `CLIENT_FRACTION`
* `N_EPOCH_PER_CLIENT`
* `BATCH_SIZE`
* `CLIENT_LEARNING_RATE`
* `SCAFFOLD`

The other parameters are for more advanced techniques. Check out the code to see their description.