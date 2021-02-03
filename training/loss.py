import torchvision
import torch.nn as nn
import torch.tensor as Tensor


class VGGPerceptual(nn.Module):
    def __init__(self):
        super(VGGPerceptual, self).__init__()
        self.validation_model = torchvision.models.vgg16(pretrained=True).cuda()
        self.validation_model.classifier = nn.Identity()
        self.validation_model.eval()
        self.loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l1 = self.loss(input, target)
        l1_features = self.loss(self.validation_model(input), self.validation_model(target))
        return l1 + l1_features
