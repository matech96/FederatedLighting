from comet_ml.exceptions import InterruptedExperiment
from comet_ml import Experiment

import os
import random
import copy
from pathlib import Path
from collections import Iterable, deque
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Callable
import logging

from statistics import mean
import numpy as np
from sklearn.metrics import confusion_matrix
from pydantic import BaseModel, validator

import torch as th
import torch.nn as nn

from torchutils.multipointer import (
    commulative_avg_model_state_dicts,
    commulative_avg_params,
    lambda_params,
    lambda2_params,
)
from mutil.ElapsedTime import ElapsedTime

from FLF.TorchOptRepo import TorchOptRepo
from FLF.TorchClient import TorchClient
from FLF.BreakedTrainingExcpetion import ToLargeLearningRateExcpetion


class FLFConfig(BaseModel):
    def flatten(self):
        res = copy.copy(self.__dict__)
        to_flatten = [k for k in res.keys() if k.endswith("_ARGS")]
        for k in to_flatten:
            poped = res.pop(k)
            for pk, pv in poped.items():
                new_key = f"{k[:-5]}_{pk}".upper()
                if isinstance(pv, Iterable) and not isinstance(pv, str):
                    for i, pvi in enumerate(pv):
                        res[f"{new_key}_{i}"] = pvi
                else:
                    res[new_key] = pv
        return res


class TorchFederatedLearnerConfig(FLFConfig):
    class Config:
        validate_assignment = True

    TARGET_ACC: float = None  # The training stopps when the test accuracy is higher, than this value.
    MAX_ROUNDS: int = 10  # The maximum number of round for training.
    N_CLIENTS: int = 2  # The number of clients to participate in a round.
    CLIENT_FRACTION: float = 1.0  # The fration of clients to participate in 1 round. Muss be between 0 and 1. 0 means selecting 1 client.
    N_EPOCH_PER_CLIENT: int = 1  # The number of epoch to train on the client before sync.
    BATCH_SIZE: int = 64  # Batch size. If set to sys.maxsize, the epoch is processed in a single batch.
    CLIENT_LEARNING_RATE: float = 0.01  # Learning rate for the client optimizer.
    SERVER_LEARNING_RATE: float = None  # Learning rate for the server optimizer. If none, the same value is used as CLIENT_LEARNING_RATE.
    SEED: int = None  # The seed.
    CLIENT_OPT: str = "SGD"  # The optimizer used by the client.
    CLIENT_OPT_L2: float = 0  # Weight decay used by the client.
    CLIENT_OPT_ARGS: Dict = {}  # Extra arguments for the client optimizer
    CLIENT_OPT_STRATEGY: str = "reinit"  # The optimizer sync strategy. Options are:
    # reinit: reinitializes the optimizer in every round
    # nothing: leavs the optimizer intect
    # avg: averages the optimizer states in every round
    CLIENT_OPT_STRATEGY_UNITL: int = None  # The CLIENT_OPT_STRATEGY is set to "reinit" after this many rounds.
    CLIENT_SGD_LEARNING_RATE: float = 0.01  # The learining rate used by SGD after the lient strategy was turned off.
    SERVER_OPT: str = None  # The optimizer used on the server.
    SERVER_OPT_ARGS: Dict = {}  # Extra arguments for the server optimizer
    SCAFFOLD: bool = False  # If true it uses SCAFFOLD, as described in arXiv:1910.06378

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


class TorchFederatedLearnerTechnicalConfig(FLFConfig):
    STORE_OPT_ON_DISK: bool = True  # If true the optimization parameters are stored on the disk between training for CLIENT_OPT_STRATEGY "nothing". This increases training time, but reduces RAM requirement. If false, it's in the RAM.
    STORE_MODEL_IN_RAM: bool = True  # If true the model is removed from the VRAM after the client has finished training. This increases training time, but reduces VRAM requirement. If false, it's kept there for the hole training.
    DL_N_WORKER: int = 0  # DataLoader: number of workers
    PIN_MEMORY: bool = True  # DataLoader: pin_memory
    HIST_SAMPLE: int = 0  # Number of sample per layer for weight histogram.
    SAVE_CHP_INTERVALL: int = 100  # Save the weights of the model to disk after this many rounds.
    BREAK_ROUND: int = None  # If the prediction is still random at this point the training is stooped. This number has to contain EVAL_ROUND.
    EVAL_ROUND: int = 1  # The number off rounds between evaluation.
    TEST_LAST: int = 0  # Test all rounds in the last few rounds, "few" equals this number.

    def check(self):  # TODO make this nicer
        assert (
            self.BREAK_ROUND % self.EVAL_ROUND == 0
        ), "BREAK_ROUND has to contain EVAL_ROUND."


class TorchFederatedLearner(ABC):
    tmp_dir = Path("tmp")

    def __init__(
        self,
        experiment: Experiment,
        config: TorchFederatedLearnerConfig,
        config_technical: TorchFederatedLearnerTechnicalConfig,
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
        self.experiment.log_parameters(self.config.flatten())
        self.config_technical = config_technical
        self.config_technical.check()
        self.experiment.log_parameters(self.config_technical.flatten())
        self.PATH = self.tmp_dir / f"{self.experiment.id}_checkpoints"
        os.makedirs(self.PATH, exist_ok=True)

        model_cls = self.get_model_cls()
        self.model = model_cls().to(self.device)
        if self.config.SERVER_OPT is not None:
            self.server_opt = TorchOptRepo.name2cls(self.config.SERVER_OPT)(
                self.model.parameters(),
                lr=self.config.SERVER_LEARNING_RATE,
                **self.config.SERVER_OPT_ARGS,
            )
        else:
            self.server_opt = None
        self.avg_opt_state = None
        if self.config.SCAFFOLD:
            self.c = lambda_params(self.model.parameters(), th.zeros_like)
            logging.info("SCAFFOLD: server c initialized")

        self.train_loader_list, self.test_loader, self.random_acc = self.load_data()
        self.n_train_batches = len(self.train_loader_list[0])
        logging.info(f"Number of training batches: {self.n_train_batches}")

        TorchClient.reset_ID_counter()
        self.clients = [
            TorchClient(
                self,
                model_cls=model_cls,
                is_keep_model_on_gpu=not self.config_technical.STORE_MODEL_IN_RAM,
                is_store_opt_on_disk=self.config_technical.STORE_OPT_ON_DISK,
                loss=self.get_loss(),
                dataloader=loader,
                device=self.device,
                opt_cls=TorchOptRepo.name2cls(self.config.CLIENT_OPT),
                opt_cls_param={
                    "lr": self.config.CLIENT_LEARNING_RATE,
                    "weight_decay": self.config.CLIENT_OPT_L2,
                },
                is_maintaine_opt_state=config.CLIENT_OPT_STRATEGY == "nothing",
                exp_id=experiment.id,
                is_scaffold=self.config.SCAFFOLD,
            )
            for loader in self.train_loader_list
        ]

    @abstractmethod
    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader, float]:
        """Loads the data.

        Returns:
            Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader] -- [The first element is the training set, the second is the test set, the third is the accuracy form random predictor]
        """
        pass

    @abstractmethod
    def get_model_cls(self) -> Callable[[], nn.Module]:
        """Returns the model to be trained.

        Returns:
            nn.Module -- The instance of the model.
        """
        pass

    @abstractmethod
    def get_loss(self, **kwargs) -> nn.Module:
        """Returns the loss function.

        Returns:
            nn.Module -- The loss function.
        """
        pass

    def train(self) -> None:
        """Runs the federated training, reports to comet.ml and runs an evaluations.

        Raises:
            ToLargeLearningRateExcpetion: Raised, when the learning rate is probably too high.

        Returns:
            None -- No return value.
        """
        lastaccs = deque(maxlen=self.config_technical.TEST_LAST)
        try:
            timer = ElapsedTime("Training")
            with timer:
                for curr_round in range(self.config.MAX_ROUNDS):
                    self.experiment.log_parameter("curr_round", curr_round)
                    # TODO if current round is larger than until
                    #   __switch_to_sgd
                    if self.config.CLIENT_OPT_STRATEGY_UNITL < curr_round:
                        self.__switch_to_sgd(self.config.CLIENT_SGD_LEARNING_RATE)
                    self.__train_one_round(curr_round)
                    is_last_testing = (
                        self.config.MAX_ROUNDS - curr_round
                    ) <= self.config_technical.TEST_LAST
                    if (curr_round % self.config_technical.EVAL_ROUND == 0) or (
                        is_last_testing
                    ):
                        self.__log("evaluation ...")
                        metrics = self.test(self.test_loader)
                        self.__log("evaluated ...")
                        self.__log("logging ...")

                        self.log_metric(
                            metrics, curr_round,
                        )
                        self.log_hist(curr_round)

                        test_acc = metrics["test_acc"]
                        lastaccs.append(test_acc)
                        if self.__is_achieved_target(test_acc):
                            break
                        if self.__is_unable_to_learn(curr_round, test_acc):
                            raise ToLargeLearningRateExcpetion()
                    if (self.config_technical.SAVE_CHP_INTERVALL is not None) and (
                        self.config_technical.SAVE_CHP_INTERVALL % (curr_round + 1) == 0
                    ):
                        th.save(self.model.state_dict(), self.PATH / f"{curr_round}.pt")
                    self.__log("logged")
                last_avg_acc = mean(lastaccs)
                self.log_metric(
                    {"last_avg_acc": last_avg_acc}, curr_round,
                )
            self.experiment.log_metric("elapsed_time_ms", timer.elapsed_time_ms)
        except InterruptedExperiment:
            pass

        th.save(self.model.state_dict(), "state_dict.pt")
        self.experiment.log_model(
            type(self).__name__, "state_dict.pt",
        )

    def __switch_to_sgd(self):
        # TODO call switch on all clients
        for client in self.clients:
            client.switch_to_sgd()
        # set strat to "reinit"
        self.config.CLIENT_OPT_STRATEGY = "reinit"
        # set client_opt to sgd
        self.config.CLIENT_OPT = "SGD"

    def __train_one_round(self, curr_round: int):
        self.model.train()

        client_sample = self.__select_clients()

        comm_avg_model_state = None
        comm_avg_opt_state = None
        if self.config.SCAFFOLD:
            comm_c = None
            logging.info("SCAFFOLD: comm_c initialized")

        with ElapsedTime("Training clients"):
            for i, client in enumerate(client_sample):
                client.set_model(self.model.state_dict())
                if (self.config.CLIENT_OPT_STRATEGY == "avg") and (
                    self.avg_opt_state is not None
                ):
                    client.set_opt_state(self.avg_opt_state)

                if self.config.SCAFFOLD:
                    client.set_server_c(self.c)

                if self.config.SCAFFOLD:
                    model_state, opt_state, c = client.train_round(
                        self.config.N_EPOCH_PER_CLIENT, curr_round
                    )
                    comm_c = commulative_avg_params(comm_c, c, i)
                    logging.info("SCAFFOLD: c received from client")
                else:
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

        with ElapsedTime("Setting gradients"):
            if self.server_opt is not None:
                self.server_opt.zero_grad()
                server_opt_state = self.server_opt.state_dict()
                self.__set_model_grads(comm_avg_model_state)
                self.server_opt = TorchOptRepo.name2cls(self.config.SERVER_OPT)(
                    self.model.parameters(),
                    lr=self.config.SERVER_LEARNING_RATE,
                    **self.config.SERVER_OPT_ARGS,
                )
                self.server_opt.load_state_dict(server_opt_state)
                self.server_opt.step()
            else:
                self.__log("setting avg model state")
                self.model.load_state_dict(comm_avg_model_state)
            if self.config.SCAFFOLD:
                self.c = lambda2_params(
                    self.c,
                    comm_c,
                    lambda a, b: a + (b * len(client_sample) / self.config.N_CLIENTS),
                )
                logging.info("SCAFFOLD: server c updated")

        self.avg_opt_state = comm_avg_opt_state
        comm_avg_opt_state = None

    def __select_clients(self):
        client_sample = random.sample(
            self.clients, max(1, int(len(self.clients) * self.config.CLIENT_FRACTION))
        )
        logging.info(f"Selected {len(client_sample)} clients in this round.")
        return client_sample

    def __set_model_grads(self, new_state):
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

    def __is_unable_to_learn(self, round, last100_avg_acc):
        return (
            (self.config_technical.BREAK_ROUND is not None)
            and (self.config_technical.BREAK_ROUND == round)
            and (abs(last100_avg_acc - self.random_acc) < 1e-6)
        )

    def __is_achieved_target(self, test_acc):
        return (self.config.TARGET_ACC is not None) and (
            test_acc > self.config.TARGET_ACC
        )

    def test(self, test_loader: th.utils.data.DataLoader) -> Dict[str, float]:
        """Tests the model on the provided dataset.

        Arguments:
            test_loader {th.utils.data.DataLoader} -- Provides the test dataset

        Returns:
            Dict[str, float] -- The meassured metrics
        """
        test_model = self.model
        test_model.to(self.device)
        test_model.eval()
        test_loss = 0
        correct = 0
        total_confusion_matrix = None

        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = test_model(data)
                test_loss += self.get_loss(reduction="sum")(
                    output, target
                ).item()  # sum up batch loss
                pred = output.argmax(
                    1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                cm = confusion_matrix(
                    target.cpu(), pred.cpu(), labels=range(output.shape[1])
                )
                if total_confusion_matrix is None:
                    total_confusion_matrix = cm
                else:
                    total_confusion_matrix += cm

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "confusion_matrix": total_confusion_matrix,
        }

    def log_client_step(
        self,
        loss: float,
        client_id: str,
        curr_round: int,
        curr_epoch: int,
        curr_batch: int,
    ):
        """A client can call this to log it's progress.

        Arguments:
            loss {float} -- The training loss
            client_id {str} -- The id of the client
            curr_round {int} -- Current round of the training
            curr_epoch {int} -- Current epoch on the client
            curr_batch {int} -- Current batch on the client
        """
        step = (
            (curr_round * self.config.N_EPOCH_PER_CLIENT) + curr_epoch
        ) * self.n_train_batches + curr_batch
        logging.info(
            f"R: {curr_round:4} E: {curr_epoch:4} B: {curr_batch:4} (S: {step} C: {client_id}) Training loss: {loss}"
        )
        self.experiment.log_metric(
            f"{client_id}_train_loss", loss, step=step, epoch=curr_round
        )

    def log_metric(self, metrics: Dict[str, float], curr_round: int):
        self.experiment.log_parameter(
            "TOTAL_EPOCH", curr_round,
        )

        if "confusion_matrix" in metrics.keys():
            self.experiment.log_confusion_matrix(
                matrix=metrics.pop("confusion_matrix"), step=curr_round
            )
        for name, value in metrics.items():
            nice_value = 100 * value if name.endswith("_acc") else value
            self.experiment.log_metric(name, nice_value, step=curr_round)
            logging.info(f"{name}: {nice_value}")

    def log_hist(self, curr_round: int):
        sample = self.config_technical.HIST_SAMPLE
        if sample == 0:
            return

        for k, v in self.model.state_dict().items():
            if th.numel(v) > sample:
                v = np.random.choice(v.flatten(), sample)
            self.experiment.log_histogram_3d(v, k, step=curr_round)

    def __log(self, m):
        logging.info(f"Server: {m}")
