import torch
import torch.tensor as Tensor


class PSNR:
    """Peak Signal to Noise Ratio
    gt should be in range [-1, 1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(outputs: Tensor, gt: Tensor) -> Tensor:
        with torch.no_grad():
            mse = torch.mean((torch.clamp(outputs, min=-1, max=1) - gt) ** 2)
            return 10 * torch.log10(4.0 / mse)
