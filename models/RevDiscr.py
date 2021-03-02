import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor


class RevDiscr(nn.Module):
    def __init__(self, num_channels: int, num_features: int, ):
        super(RevDiscr, self).__init__()
        self.in_channels = num_channels
        self.num_filters = num_features
        self.stride = 2
        self.kernel_size = 3
        self.drop = nn.Dropout2d(0.2)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_filters,
                               kernel_size=self.kernel_size, padding=1,
                               stride=self.stride)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters,
                               out_channels=self.num_filters,
                               kernel_size=self.kernel_size, padding=1,
                               stride=self.stride)
        self.conv3 = nn.Conv2d(in_channels=self.num_filters,
                               out_channels=self.num_filters,
                               kernel_size=self.kernel_size, padding=1,
                               stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.num_filters,
                               out_channels=self.num_filters,
                               kernel_size=self.kernel_size, padding=1,
                               stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.num_filters,
                               out_channels=self.num_filters,
                               kernel_size=self.kernel_size, padding=1,
                               stride=self.stride)
        self.conv6 = nn.Conv2d(in_channels=self.num_filters,
                               out_channels=1,
                               kernel_size=self.kernel_size, padding=1,
                               stride=self.stride)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = F.relu(self.conv4(x))
        x = self.drop(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def requires_grad(self, requires_grad: bool=True):
        if requires_grad:
            self.train()
            for param in self.parameters():
                param.requires_grad = True
        else:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
