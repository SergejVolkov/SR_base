import torch
import torch.nn as nn
# import torch.tensor as Tensor
import dl_modules.dataset as ds


def save(epoch_idx: int, best_accuracy: float, gen_model: nn.Module,
         dis_model: nn.Module, gen_opt: torch.optim.Optimizer, dis_opt: torch.optim.Optimizer):
    checkpoint = {
        'epoch': epoch_idx,
        'best_acc': best_accuracy,
        'generator': gen_model.state_dict(),
        'discriminator': dis_model.state_dict(),
        'gen_optimizer': gen_opt.state_dict(),
        'dis_optimizer': dis_opt.state_dict()
    }
    torch.save(checkpoint, ds.SAVE_DIR + 'weights/checkpoint')


def load(gen_model: nn.Module, dis_model: nn.Module,
         gen_opt: torch.optim.Optimizer, dis_opt: torch.optim.Optimizer):
    checkpoint = torch.load(ds.SAVE_DIR + 'weights/checkpoint')
    gen_model.load_state_dict(checkpoint['generator'])
    dis_model.load_state_dict(checkpoint['discriminator'])
    gen_opt.load_state_dict(checkpoint['gen_optimizer'])
    dis_opt.load_state_dict(checkpoint['dis_optimizer'])