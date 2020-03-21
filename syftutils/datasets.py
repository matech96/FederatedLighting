import numpy as np
import torch as th


def get_dataset_items_at(ds, idx):
    data = th.stack([ds[i][0] for i in idx])
    target = ds.targets[idx]
    return data, target
