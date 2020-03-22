from collections import defaultdict
import torch as th
import suft as sy


def get_dataset_items_at(ds, idx):
    data = th.stack([ds[i][0] for i in idx])
    target = ds.targets[idx]
    return data, target


class ClientBatchIter:
    def __init__(self, federated_data_loader: sy.FederatedDataLoader):
        self.federated_data_loader = federated_data_loader

    def __iter__(self):
        self.iter = iter(self.federated_data_loader)
        self.client_curr_batch = defaultdict(lambda: 0)
        return self

    def __next__(self):
        data, target = next(self.iter)
        client_id = data.location
        curr_batch = self.client_curr_batch[client_id]
        self.client_curr_batch[client_id] += 1
        return curr_batch, (data, target)
