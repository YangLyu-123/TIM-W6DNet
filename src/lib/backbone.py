import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1) # bs x 64 x 32 x 32
        c3 = self.layer2(c2) # bs x 128 x 16 x 16
        c4 = self.layer3(c3) # bs x 256 x 8 x 8
        c5 = self.layer4(c4) # bs x 512 x 4 x 4
        feats = self.avgpool(c5) # bs x 512 x 1 x 1
        return [c2,c3,c4,c5,feats]