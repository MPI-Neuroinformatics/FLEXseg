"""This module provides functionality to work with gpu."""

import torch
import torch.nn as nn


def set_up_cuda() -> torch.device:
    "Check if GPU is available on the system and use it."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def parallelize_model(model: nn.Module, device_id: int,
                      batch_size: int) -> nn.Module:
    """Parallelize model if possible."""
    # model = model.cuda(0)
    torch.cuda.set_device(device_id)
    if (torch.cuda.device_count() >= device_id
            and batch_size >= torch.cuda.device_count()):
        model = nn.parallel.DistributedDataParallel(model, device_id)

    return model