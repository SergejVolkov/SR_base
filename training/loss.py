import torch
import torch.nn as nn
import torch.tensor as Tensor


class VGGPerceptual(nn.Module):
    def __init__(self):
        super(VGGPerceptual, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_exp_sum = torch.log(torch.exp(input).sum(1))
        item_loss = torch.zeros(target.shape)
        for i in range(target.shape[0]):
            item_loss[i] = -input[i][target[i]] + log_exp_sum[i]
        average_batch_loss = item_loss.sum() / target.shape[0]
        return average_batch_loss
