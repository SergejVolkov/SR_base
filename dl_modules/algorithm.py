import torch.optim as optim
import torch.nn as nn
from dl_modules.metric.psnr import PSNR
from dl_modules.metric.ssim import SSIM
from lpips import LPIPS
from dl_modules.loss import VGGPerceptual, LSGANGenLoss, \
    LSGANDisLoss, LSGANDisFakeLoss, LSGANDisRealLoss


gen_opt_state_dict = None
dis_opt_state_dict = None

gan_loss_coeff = 0.05
init_gen_lr = 0.001
dis_lr = 0.0001

lpips = None
ssim = None
psnr = None


def get_lpips() -> nn.Module:
    global lpips
    if lpips is None:
        lpips = LPIPS(verbose=False)
    return lpips


def get_ssim() -> nn.Module:
    global ssim
    if ssim is None:
        ssim = SSIM()
    return ssim


def get_psnr() -> nn.Module:
    global psnr
    if psnr is None:
        psnr = PSNR()
    return psnr


def get_super_loss() -> nn.Module:
    return VGGPerceptual(l1_coeff=0.01, features_coeff=1)


def get_gen_loss() -> nn.Module:
    return LSGANGenLoss()


def get_dis_loss() -> nn.Module:
    return LSGANDisLoss()


def get_dis_fake_loss() -> nn.Module:
    return LSGANDisFakeLoss()


def get_dis_real_loss() -> nn.Module:
    return LSGANDisRealLoss()


def get_gen_optimizer(net: nn.Module) -> optim.Optimizer:
    global gen_opt_state_dict
    optimizer = optim.Adam(net.parameters(), lr=init_gen_lr, betas=(0.5, 0.999))
    if gen_opt_state_dict is not None:
        optimizer.load_state_dict(gen_opt_state_dict)
    return optimizer


def get_dis_optimizer(net: nn.Module) -> optim.Optimizer:
    global dis_opt_state_dict
    optimizer = optim.Adam(net.parameters(), lr=dis_lr, betas=(0.5, 0.999))
    if dis_opt_state_dict is not None:
        optimizer.load_state_dict(dis_opt_state_dict)
    return optimizer


def update_optimizer(optimizer: optim.Optimizer, params: tuple) -> None:
    for g in optimizer.param_groups:
        g['lr'] = params[0]
    pass
