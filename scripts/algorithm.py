import torch.optim as optim
import torch.nn as nn
from scripts.metric import PSNR
from scripts.loss import VGGPerceptual


def get_loss() -> nn.Module:
    return VGGPerceptual()


def get_metric():
    return PSNR()


def get_optimizer(net: nn.Module) -> optim.Optimizer:
    return optim.Adam(net.parameters())


def update_optimizer(optimizer: optim.Optimizer, params: tuple) -> None:
    for g in optimizer.param_groups:
        g['lr'] = params[0]
    pass
