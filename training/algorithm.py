import torch.optim as optim
import torch.nn as nn
from training.metric import PSNR


def get_loss() -> nn.Module:
    return nn.L1Loss()


def get_metric():
    return PSNR()


def get_optimizer(net: nn.Module, params: list=tuple()) -> optim.Optimizer:
    return optim.Adam(net.parameters(), lr=params[0])


def update_optimizer(optimizer: optim.Optimizer, params: list) -> None:
    for g in optimizer.param_groups:
        g['lr'] = params[0]
    pass
