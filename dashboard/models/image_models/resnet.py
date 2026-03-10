# @title ResNet  Model
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
####### resnet 18

# class ResNet18_MedMNIST(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()

#         self.model = models.resnet(weights=None)

#         # Modify first conv layer
#         self.model.conv1 = nn.Conv2d(
#             1, 64, kernel_size=3, stride=1, padding=1, bias=False
#         )

#         # Remove maxpool

#         # Keep maxpool for 64 (safe now)
#         # self.model.maxpool = nn.Identity()

#         # Modify final layer
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)
######### Resnet 8 veriosn



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, num_classes=2,in_channels=1):
        super(ResNet8, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = BasicBlock(16, 16, stride=1)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out