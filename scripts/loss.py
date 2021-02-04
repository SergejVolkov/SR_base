import torchvision
import torch.nn as nn
import torch.tensor as Tensor


class VGGPerceptual(nn.Module):
    def __init__(self):
        super(VGGPerceptual, self).__init__()
        self.validation_model = torchvision.models.vgg16(pretrained=True).cuda()
        # Remove classifier part
        self.validation_model.classifier = nn.Identity()
        # Remove layers with deep features
        self.validation_model.features = nn.Sequential(*self.validation_model.features[:22])
        # Freeze model
        self.validation_model.eval()
        # Create L1 loss for measuring distance
        self.loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l1 = self.loss(input, target)
        l1_features = self.loss(self.validation_model(input), self.validation_model(target))
        return l1 + l1_features
