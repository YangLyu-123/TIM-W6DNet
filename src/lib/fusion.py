import torch
import torch.nn as nn

class FusionRes(nn.Module):
    def __init__(self):
        super(FusionRes, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.D = nn.Sequential(
            nn.Linear(1984, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, img_feat_list, pcd_feat_list):
        c2, c3, c4, c5, _ = img_feat_list
        p1, _ = pcd_feat_list

        img_feat = torch.cat(
            [
                self.avgpool(c2),
                self.avgpool(c3),
                self.avgpool(c4),
                self.avgpool(c5)
            ], dim=1
        )
        img_feat = img_feat.squeeze(-1).squeeze(-1)
        f1_feat = torch.cat([img_feat, p1], dim=1) # F1 block
        d_feat = self.D(f1_feat) # D block
        return f1_feat, d_feat