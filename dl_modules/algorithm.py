import torch.optim as optim
import torch.nn as nn
from dl_modules.metric import PSNR
from dl_modules.loss import VGGPerceptual, LSGANDisLoss, LSGANGenLoss


gen_opt_state_dict = None
dis_opt_state_dict = None

gan_loss_coeff = 0.1
init_gen_lr = 0.001
init_dis_lr = 0.0001


def get_super_loss() -> nn.Module:
    return VGGPerceptual(l1_coeff=0.01, features_coeff=1)


def get_gen_loss() -> nn.Module:
    return LSGANGenLoss()


def get_dis_loss() -> nn.Module:
    return LSGANDisLoss()


def get_metric():
    return PSNR()


def get_gen_optimizer(net: nn.Module) -> optim.Optimizer:
    global gen_opt_state_dict
    optimizer = optim.Adam(net.parameters(), lr=init_gen_lr, betas=(0.5, 0.999))
    if gen_opt_state_dict is not None:
        optimizer.load_state_dict(gen_opt_state_dict)
    return optimizer


def get_dis_optimizer(net: nn.Module) -> optim.Optimizer:
    global dis_opt_state_dict
    optimizer = optim.Adam(net.parameters(), lr=init_dis_lr, betas=(0.5, 0.999))
    if dis_opt_state_dict is not None:
        optimizer.load_state_dict(dis_opt_state_dict)
    return optimizer


def update_optimizer(optimizer: optim.Optimizer, params: tuple) -> None:
    for g in optimizer.param_groups:
        g['lr'] = params[0]
    pass
