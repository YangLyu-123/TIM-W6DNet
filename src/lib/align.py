import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from torch.autograd import Variable, Function
import numpy as np

class AlignModule(nn.Module): # ResBlock lambda
    def __init__(self, in_channels=256):
        super(AlignModule, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out