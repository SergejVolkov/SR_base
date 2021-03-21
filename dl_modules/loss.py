import math
import torch
import torchvision
import torch.nn as nn
import torch.tensor as Tensor


class VGGPerceptual(nn.Module):
    def __init__(self, l1_coeff: float=1.0, features_coeff: float=1.0,
                 edge_coeff: float=1.0):
        super(VGGPerceptual, self).__init__()
        # Set params
        self.a = l1_coeff
        self.b = features_coeff
        self.c = edge_coeff
        # Get pretrained VGG model
        self.validation_model = torchvision.models.vgg16(pretrained=True).cuda()
        # Remove classifier part
        self.validation_model.classifier = nn.Identity()
        # Remove layers with deep features
        self.validation_model.features = nn.Sequential(*self.validation_model.features[:22])
        # Freeze model
        self.validation_model.eval()
        for param in self.validation_model.parameters():
            param.requires_grad = False
        # Create L1 loss and Edge loss
        self.loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l1 = self.loss(input, target)
        l1_features = self.loss(self.validation_model(input), self.validation_model(target))
        edge = self.edge_loss(input, target)
        return self.a * l1 + self.b * l1_features + self.c * edge

    def myto(self, device: torch.device):
        self.edge_loss.myto(device)


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

        channels = 3
        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32,
            requires_grad=False
        )
        sobel_x_kernel = sobel_x.repeat(channels, 1, 1, 1)
        sobel_x_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                   kernel_size=3, groups=channels, bias=False)
        sobel_x_filter.weight.data = sobel_x_kernel
        sobel_x_filter.weight.requires_grad = False
        self.sobel_x_filter = sobel_x_filter

        sobel_y = torch.transpose(sobel_x, 0, 1)
        sobel_y_kernel = sobel_y.repeat(channels, 1, 1, 1)
        sobel_y_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                   kernel_size=3, groups=channels, bias=False)
        sobel_y_filter.weight.data = sobel_y_kernel
        sobel_y_filter.weight.requires_grad = False
        self.sobel_y_filter = sobel_y_filter

        kernel_size = 3
        sigma = 3

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * \
            torch.exp(
                -torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance)
            )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        self.gaussian = gaussian_filter

        self.loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        denoised_input = self.gaussian(input)
        denoised_target = self.gaussian(target)
        edges_map_input = (self.sobel_x_filter(denoised_input) ** 2 +
                           self.sobel_y_filter(denoised_input) ** 2) ** (1 / 2)
        edges_map_target = (self.sobel_x_filter(denoised_target) ** 2 +
                            self.sobel_y_filter(denoised_target) ** 2) ** (1 / 2)
        return self.loss(edges_map_input, edges_map_target)

    def myto(self, device: torch.device):
        self.gaussian_filter.weight.data.to(device)
        self.sobel_x_filter.weight.data.to(device)
        self.sobel_y_filter.weight.data.to(device)


class LSGANGenLoss(nn.Module):
    def __init__(self):
        super(LSGANGenLoss, self).__init__()

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        return torch.mean((fake - 1) ** 2)


class LSGANDisLoss(nn.Module):
    def __init__(self):
        super(LSGANDisLoss, self).__init__()

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        return torch.mean(fake ** 2 + (real - 1) ** 2)


class LSGANDisFakeLoss(nn.Module):
    def __init__(self):
        super(LSGANDisFakeLoss, self).__init__()

    def forward(self, fake: Tensor) -> Tensor:
        return torch.mean(fake ** 2)


class LSGANDisRealLoss(nn.Module):
    def __init__(self):
        super(LSGANDisRealLoss, self).__init__()

    def forward(self, real: Tensor) -> Tensor:
        return torch.mean((real - 1) ** 2)
