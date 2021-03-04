import torch
import torchvision
import torch.nn as nn
import torch.tensor as Tensor


class VGGPerceptual(nn.Module):
    def __init__(self, l1_coeff: float=1.0, features_coeff: float=1.0):
        super(VGGPerceptual, self).__init__()
        # Set params
        self.a = l1_coeff
        self.b = features_coeff
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
        # Create L1 loss for measuring distance
        self.loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l1 = self.loss(input, target)
        l1_features = self.loss(self.validation_model(input), self.validation_model(target))
        return self.a * l1 + self.b * l1_features


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
