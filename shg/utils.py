"""Shared model-building components."""
import random
from typing import Optional

import numpy as np
import torch
from torch import nn


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
    
def fix_outputs_by_crystal_type(outputs, crystal_type):
    if crystal_type == "m":
        outputs[0, 3] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 5] = 0.0
        outputs[0, 4] = outputs[2, 0]
        outputs[2, 4] = outputs[0, 2]
        outputs[1, 5] = outputs[0, 1]
    elif crystal_type == "2":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 4] = 0.0
        outputs[0, 5] = outputs[1, 0]
        outputs[1, 4] = outputs[0, 3]
        outputs[2, 5] = outputs[0, 3]
        outputs[1, 3] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 0] = 0.0
        outputs[2, 1] = 0.0
        outputs[2, 2] = 0.0
        outputs[2, 4] = 0.0
    elif crystal_type == "mm2":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
        outputs[0, 4] = outputs[2, 0]
        outputs[1, 3] = outputs[2, 1]
    elif crystal_type == "3m":
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
        outputs[0, 4] = outputs[2, 0]
        outputs[1, 5] = outputs[0, 1]
    elif crystal_type == "32":
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 4] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 3] = 0.0
        outputs[1, 4] = 0.0
        outputs[2, :] = 0.0
    elif crystal_type == "4-2m":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
      #  outputs[0, 4] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 2] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = outputs[0, 3]
        outputs[1, 4] = outputs[0, 3]
    elif crystal_type == "6mm":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
        outputs[0, 4] = outputs[2, 0]
        outputs[1, 3] = outputs[2, 0]

    elif crystal_type == "3":
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
    elif crystal_type == "222":
       # print(outputs)
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 4] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 3] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 0] = 0.0
        outputs[2, 1] = 0.0
        outputs[2, 2] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[1, 4] = outputs[0, 3]
        outputs[2, 5] = outputs[0, 3]
        #print(outputs)
    elif crystal_type == "6-m2":
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 4] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 3] = 0.0
        outputs[1, 4] = 0.0
        outputs[2, :] = 0.0
    elif crystal_type == "6":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
    elif crystal_type == "4-" or crystal_type == "-4":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = outputs[0, 3]
        outputs[1, 4] = outputs[0, 3]
    elif crystal_type == "1":
        return outputs
    elif crystal_type == "4mm":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
        outputs[0, 4] = outputs[2, 0]
        outputs[1, 3] = outputs[2, 0]
    elif crystal_type == "4-3m":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 4] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 3] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 0] = 0.0
        outputs[2, 1] = 0.0
        outputs[2, 2] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
    elif crystal_type == "23":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 4] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 3] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 0] = 0.0
        outputs[2, 1] = 0.0
        outputs[2, 2] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
    elif crystal_type == "mmm":
        outputs[:, :] = 0.0
    elif crystal_type == "4":
        outputs[0, 0] = 0.0
        outputs[0, 1] = 0.0
        outputs[0, 2] = 0.0
        outputs[0, 3] = 0.0
        outputs[0, 5] = 0.0
        outputs[1, 0] = 0.0
        outputs[1, 1] = 0.0
        outputs[1, 2] = 0.0
        outputs[1, 4] = 0.0
        outputs[1, 5] = 0.0
        outputs[2, 3] = 0.0
        outputs[2, 4] = 0.0
        outputs[2, 5] = 0.0
    return outputs