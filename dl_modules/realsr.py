import os
import random
import numpy as np
import torch
import torch.tensor as Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset as BaseDataset
# from scipy.ndimage import measurements, interpolation


class Kernels:
    def __init__(self, kernels_dir, scale, count=None):
        self.ids = os.listdir(kernels_dir)
        random.seed(7)
        random.shuffle(self.ids)
        random.seed()
        self.kernels_fps = [os.path.join(kernels_dir, kernel_id) for kernel_id in self.ids]
        self.scale = scale
        self.count = len(self.ids)
        if count is not None and count < self.count:
            self.count = count

    def random_kernel(self):
        return self.__getitem__(random.randrange(0, self.count))

    def __getitem__(self, i: int):
        kernel = np.loadtxt(self.kernels_fps[i])

        # KernelGAN official implementation

        # # First calculate the current center of mass for the kernel
        # current_center_of_mass = measurements.center_of_mass(kernel)
        #
        # # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
        # wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * np.array([1.0, 1.0])
        # # Define the shift vector for the kernel shifting (x,y)
        # shift_vec = wanted_center_of_mass - current_center_of_mass
        # # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
        # # (biggest shift among dims + 1 for safety)
        # kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
        #
        # # Finally shift the kernel and return
        # kernel = interpolation.shift(kernel, shift_vec)

        return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return self.count


def inject_noise(image: Tensor, noise_loader: torch.utils.data.DataLoader) -> Tensor:
    noise, gt = next(iter(noise_loader))
    noise = noise.to(image.device)
    if noise_amount is not None:
        clamp = noise_amount * 2 / 255
        noise = torch.clamp(noise, -clamp, clamp)
    return torch.clamp(image + noise, -1.0, 1.0)


def apply_kernel(image: Tensor, kernel_storage: Kernels):
    unsquuezed_cnt = 0
    while len(image.shape) < 4:
        image = image.unsqueeze(0)
        unsquuezed_cnt += 1
    kernel = kernel_storage.random_kernel()
    padding = (kernel.shape[-1] - 1) // 2
    image = F.pad(image, [padding for _ in range(4)], mode='reflect')
    downscaled = torch.cat(
        [F.conv2d(image[:, i:i+1, :, :], kernel, stride=kernel_storage.scale)
         for i in range(image.shape[1])],
        dim=1
    )
    while unsquuezed_cnt > 0:
        downscaled = downscaled.squeeze(0)
        unsquuezed_cnt -= 1
    return downscaled


def get_noise_normalization() -> torch.nn.Module:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (noise_mean / 255, noise_mean / 255, noise_mean / 255),
            (0.5, 0.5, 0.5)
        )
    ])


kernel_count = 26
noise_amount = 20
noise_mean = 128
