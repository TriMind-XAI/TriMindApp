# @title Simple CNN Model
import torch
import torch.nn as nn
import torch.nn.functional as F
####### 28 size
class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BetterCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(BetterCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 28x28
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 7x7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> 3x3

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

####### 64 size
# class SmallCNN(nn.Module):
#     def __init__(self, in_channels=1, num_classes=2):
#         super(SmallCNN, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

#         self.fc1 = nn.Linear(32 * 4 * 4, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))

#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)

#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

