from comet_ml.exceptions import InterruptedExperiment
from comet_ml import Experiment

import random
import copy
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Callable
import logging

import numpy as np
from pydantic import BaseModel, validator

import torch as th
import torch.nn as nn

from syftutils.multipointer import commulative_avg_model_state_dicts
from mutil.ElapsedTime import ElapsedTime

from FLF.TorchOptRepo import TorchOptRepo
from FLF.TorchClient import TorchClient


class TorchFederatedLearnerConfig(BaseModel):
    class Config:
        validate_assignment = True

    TARGET_ACC: float = 0.99  # The training stopps when the test accuracy is higher, than this value.
    MAX_ROUNDS: int = 10  # The maximum number of round for training.
    N_CLIENTS: int = 2  # The number of clients to participate in a round.
    CLIENT_FRACTION: float = 1.0  # The fration of clients to participate in 1 round. Muss be between 0 and 1. 0 means selecting 1 client.
    N_EPOCH_PER_CLIENT: int = 1  # The number of epoch to train on the client before sync.
    BATCH_SIZE: int = 64  # Batch size. If set to sys.maxsize, the epoch is processed in a single batch.
    CLIENT_LEARNING_RATE: float = 0.01  # Learning rate for the client optimizer.
    SERVER_LEARNING_RATE: float = None  # Learning rate for the server optimizer. If none, the same value is used as CLIENT_LEARNING_RATE.
    DL_N_WORKER: int = 4  # Syft.FederatedDataLoader: number of workers
    SEED: int = None  # The seed.
    CLIENT_OPT: str = "SGD"  # The optimizer used by the client.
    CLIENT_OPT_STRATEGY: str = "reinit"  # The optimizer sync strategy. Options are:
    # reinit: reinitializes the optimizer in every round
    # nothing: leavs the optimizer intect
    # avg: averages the optimizer states in every round
    SERVER_OPT: str = None  # The optimizer used on the server.

    @staticmethod
    def __percentage_validator(value: float) -> None:
        if (0.0 > value) or (value > 1.0):
            raise ValueError("CLIENT_FRACTION muss be between 0 and 1.")
        else:
            return value

    _val_CLIENT_FRACTION = validator("CLIENT_FRACTION", allow_reuse=True)(
        __percentage_validator.__func__
    )
    _val_TARGET_ACC = validator("TARGET_ACC", allow_reuse=True)(
        __percentage_validator.__func__
    )

    def set_defaults(self):  # TODO make this nicer
        if self.SERVER_LEARNING_RATE is None:
            self.SERVER_LEARNING_RATE = self.CLIENT_LEARNING_RATE


class TorchFederatedLearner(ABC):
    def __init__(
        self, experiment: Experiment, config: TorchFederatedLearnerConfig
    ) -> None:
        """Initialises the training.

        Arguments:
            experiment {Experiment} -- Comet.ml experiment object for online logging.
            config {TorchFederatedLearnerConfig} -- Training configuration description.
        """
        super().__init__()
        if config.SEED is not None:
            random.seed(config.SEED)
            np.random.seed(config.SEED)
            th.manual_seed(config.SEED)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False

        self.device = "cuda"  # th.device("cuda" if th.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.config = config
        self.config.set_defaults()
        self.experiment.log_parameters(self.config.__dict__)

        model_cls, is_keep_model_on_gpu = self.get_model_cls()
        self.model = model_cls()
        if self.config.SERVER_OPT is not None:
            self.server_opt = TorchOptRepo.name2cls(self.config.SERVER_OPT)(
                self.model.parameters(), lr=self.config.SERVER_LEARNING_RATE
            )
        else:
            self.server_opt = None
        self.avg_opt_state = None

        self.train_loader_list, self.test_loader = self.load_data()
        self.n_train_batches = int(
            len(self.train_loader_list) / self.config.N_CLIENTS
        )  # TODO batch per client
        logging.info(f"Number of training batches: {self.n_train_batches}")

        TorchClient.reset_ID_counter()
        self.clients = [
            TorchClient(
                self,
                model_cls,
                is_keep_model_on_gpu,
                self.get_loss(),
                loader,
                self.device,
                TorchOptRepo.name2cls(self.config.CLIENT_OPT),
                {"lr": self.config.CLIENT_LEARNING_RATE},
                config.CLIENT_OPT_STRATEGY == "nothing",
            )
            for loader in self.train_loader_list
        ]

    @abstractmethod
    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader]:
        """Loads the data.

        Returns:
            Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader] -- [The first element is the training set, the second is the test set]
        """
        pass

    @abstractmethod
    def get_model_cls(self) -> Tuple[Callable[[], nn.Module], bool]:
        """Returns the model to be trained.

        Returns:
            nn.Module -- The instance of the model.
            bool -- If false the model will only be allocated on the gpu if the client, that it belongs to is beeing trained.
            This increases training time, but reduces GPU memory requirement.
        """
        pass

    @abstractmethod
    def get_loss(self) -> nn.Module:
        """Returns the loss function.

        Returns:
            nn.Module -- The loss function.
        """
        pass

    def train(self) -> None:
        """Runs the federated training, reports to comet.ml and runs an evaluations.

        Returns:
            None -- No return value.
        """
        try:
            with ElapsedTime("Training"):
                for round in range(self.config.MAX_ROUNDS):
                    self.experiment.log_parameter("curr_round", round)
                    self.__train_one_round(round)
                    metrics = self.test(self.test_loader)
                    self.log_test_metric(
                        metrics,
                        round * self.config.N_EPOCH_PER_CLIENT * self.n_train_batches,
                    )
                    logging.info(f'Test accuracy: {metrics["test_acc"]}')

                    if metrics["test_acc"] > self.config.TARGET_ACC:
                        break
        except InterruptedExperiment:
            pass

        # th.save(model.state_dict(), "mnist_cnn.pt")

    def __train_one_round(self, curr_round: int):
        self.model.train()

        client_sample = self.__select_clients()

        comm_avg_model_state = None
        comm_avg_opt_state = None

        for i, client in enumerate(client_sample):
            client.set_model(self.model.state_dict())
            if (self.config.CLIENT_OPT_STRATEGY == "avg") and (
                self.avg_opt_state is not None
            ):
                client.set_opt_state(self.avg_opt_state)

            model_state, opt_state = client.train_round(
                self.config.N_EPOCH_PER_CLIENT, curr_round
            )

            comm_avg_model_state = commulative_avg_model_state_dicts(
                comm_avg_model_state, model_state, i
            )

            if self.config.CLIENT_OPT_STRATEGY == "avg":
                if comm_avg_opt_state is not None:
                    comm_avg_opt_state = [
                        commulative_avg_model_state_dicts(opt_s[0], opt_s[1], i)
                        for opt_s in zip(comm_avg_opt_state, opt_state)
                    ]
                else:
                    comm_avg_opt_state = opt_state

        if self.server_opt is not None:
            self.__log("setting gradients")
            self.__set_model_grads(comm_avg_model_state)
            self.server_opt.step()
        else:
            self.__log("setting avg model state")
            self.model.load_state_dict(comm_avg_model_state)
        self.avg_opt_state = comm_avg_opt_state
        comm_avg_opt_state = None

    def __select_clients(self):
        client_sample = random.sample(
            self.clients, max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        )
        logging.info(f"Selected {len(client_sample)} clients in this round.")
        return client_sample

    def __set_model_grads(self, new_state):
        self.server_opt.zero_grad()
        new_model = copy.deepcopy(self.model)
        new_model.load_state_dict(new_state)
        with th.no_grad():
            for parameter, new_parameter in zip(
                self.model.parameters(), new_model.parameters()
            ):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model.load_state_dict(new_model_state_dict)

    def test(self, test_loader: th.utils.data.DataLoader) -> Dict[str, float]:
        test_model = copy.deepcopy(self.model)
        test_model.to(self.device)
        test_model.eval()
        test_loss = 0
        correct = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = test_model(data)
                test_loss += self.get_loss()(
                    output, target  # , reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        return {"test_loss": test_loss, "test_acc": test_acc}

    def log_client_step(
        self,
        loss: float,
        client_id: str,
        curr_round: int,
        curr_epoch: int,
        curr_batch: int,
    ):
        step = (
            (curr_round * self.config.N_EPOCH_PER_CLIENT) + curr_epoch
        ) * self.n_train_batches + curr_batch
        logging.info(
            f"R: {curr_round:4} E: {curr_epoch:4} B: {curr_batch:4} (S: {step} C: {client_id}) Training loss: {loss}"
        )
        self.experiment.log_metric(f"{client_id}_train_loss", loss, step=step)

    def log_test_metric(self, metrics: Dict[str, float], batch_num: int):
        self.experiment.log_parameter(
            "TOTAL_EPOCH",
            self.config.N_EPOCH_PER_CLIENT
            * batch_num
            / (self.config.N_EPOCH_PER_CLIENT * self.n_train_batches),
        )
        for name, value in metrics.items():
            nice_value = 100 * value if name.endswith("_acc") else value
            self.experiment.log_metric(name, nice_value, step=batch_num)
            logging.info(f"{name}: {nice_value}")

    def __log(self, m):
        logging.info(f"Server: {m}")
