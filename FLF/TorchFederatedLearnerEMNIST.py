from comet_ml import Experiment

import os
from typing import Tuple, List, Callable
import logging

from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from FFL.TorchFederatedLearner import TensorFlowFederatedLearner


class TorchFederatedLearnerEMNIST(TensorFlowFederatedLearner):
    def load_data(
        self,
    ) -> Tuple[List[th.utils.data.DataLoader], th.utils.data.DataLoader]:
        logging.info("MNIST data loading ...")
        minist_train_ds, mnist_test_ds = self.__get_mnist()
        logging.info("MNIST data loaded.")

        logging.info("Data for client is being sampled ...")
        n_training_samples = len(minist_train_ds)
        logging.info("Number of training samples: {n_training_samples}")
        if self.config.IS_IID_DATA:
            indices = np.arange(n_training_samples)
            indices = indices.reshape(self.config.N_CLIENTS, -1)
            indices = indices.tolist()
        else:
            indices = self.__distribute_data_non_IID(minist_train_ds)

        train_loader_list = []
        for idx in indices:
            sampler = th.utils.data.sampler.SubsetRandomSampler(idx)
            loader = th.utils.data.DataLoader(
                dataset=minist_train_ds,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.DL_N_WORKER,
                sampler=sampler,
            )
            train_loader_list.append(loader)
        logging.info("Data for client is sampled.")

        test_loader = th.utils.data.DataLoader(
            mnist_test_ds, batch_size=64, num_workers=self.config.DL_N_WORKER,
        )

        return train_loader_list, test_loader

    def __get_emnist_filenames(self):
        base_dir = Path('data')
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
        process_emnist_tff_set(emnist_train, base_dir / 'train')
        process_emnist_tff_set(emnist_test, base_dir / 'test')

    def __process_emnist_tff_set(emnist, base_dir):
        try:
            os.makedirs(base_dir)
        except OSError:
            return
        
        for client_id in tqdm(emnist.client_ids):
            tf_ds = emnist.create_tf_dataset_for_client(client_id)
            x = [np.expand_dims(x['pixels'], axis=0) for x in tf_ds.as_numpy_iterator()]
            x = np.stack(x)
            y = [np.expand_dims(x['label'], axis=0) for x in tf_ds.as_numpy_iterator()]
            y = np.stack(y)
            np.save(base_dir / client_id, {'x': x, 'y': y})

    def get_model_cls(self) -> Callable[[], nn.Module]:
        return Net


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
