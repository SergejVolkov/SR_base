import torch.nn.functional as F
from torch import nn


class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()
        self.eval()

    def forward(self, x):
        return F.interpolate(x, scale_factor=(2, 2), mode='bicubic')
 