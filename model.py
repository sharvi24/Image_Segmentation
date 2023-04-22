import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 7, 1)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.max_pool(x1)
        x2 = F.relu(self.conv2(x2))
        x3 = self.max_pool(x2)
        x3 = F.relu(self.conv3(x3))
        x4 = self.max_pool(x3)
        x4 = F.relu(self.conv4(x4))
        x5 = self.max_pool(x4)
        x5 = F.relu(self.conv5(x5))

        x6 = self.upsample(x5)
        x6 = F.relu(self.conv6(torch.cat([x6, x4], dim=1)))
        x7 = self.upsample(x6)
        x7 = F.relu(self.conv7(torch.cat([x7, x3], dim=1)))
        x8 = self.upsample(x7)
        x8 = F.relu(self.conv8(torch.cat([x8, x2], dim=1)))
        x9 = self.upsample(x8)
        x9 = F.relu(self.conv9(torch.cat([x9, x1], dim=1)))