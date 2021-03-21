import torch
import torch.nn as nn
# import torch.tensor as Tensor
import dl_modules.dataset as ds
import dl_modules.scheduler.exp as scheduler


def save(epoch_idx: int, best_accuracy: float, gen_model: nn.Module,
         dis_model: nn.Module, gen_opt: torch.optim.Optimizer, dis_opt: torch.optim.Optimizer):
    checkpoint = {
        'epoch': epoch_idx,
        'best_acc': best_accuracy,
        'lr': scheduler.gen_lr,
        'generator': gen_model.state_dict(),
        'discriminator': dis_model.state_dict(),
        'gen_optimizer': gen_opt.state_dict(),
        'dis_optimizer': dis_opt.state_dict()
    }
    torch.save(checkpoint, ds.SAVE_DIR + 'weights/checkpoint')


def load(gen_model: nn.Module, dis_model: nn.Module,
         gen_opt: torch.optim.Optimizer, dis_opt: torch.optim.Optimizer):
    checkpoint = torch.load(ds.SAVE_DIR + 'weights/checkpoint')
    scheduler.gen_lr = checkpoint['lr']
    gen_model.load_state_dict(checkpoint['generator'])
    dis_model.load_state_dict(checkpoint['discriminator'])
    gen_opt.load_state_dict(checkpoint['gen_optimizer'])
    dis_opt.load_state_dict(checkpoint['dis_optimizer'])


def unpack(name: str='checkpoint'):
    checkpoint = torch.load(ds.SAVE_DIR + 'weights/' + name)
    epoch_idx = checkpoint['epoch']
    valid_metric = checkpoint['best_acc']
    PATH = ds.SAVE_DIR + 'weights/'
    torch.save(checkpoint['generator'], PATH + 'gen_epoch_%04d_lpips_%.4f.pth' % (epoch_idx, valid_metric))
    torch.save(checkpoint['generator'], PATH + 'dis_epoch_%04d_lpips_%.4f.pth' % (epoch_idx, valid_metric))
