import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from torch.autograd import Variable, Function
import numpy as np
from .utils import quaternion_to_matrix, _cal_kl
from .pointnet import PointNetfeat
from .fusion import *
from .backbone import *
from .align import *

class W6DNet(nn.Module):
    def __init__(self, cfg):
        super(W6DNet, self).__init__()
        self.ind_feat = cfg.MODEL.IND_FEAT
        self.loss_term = set(cfg.LOSS.TERM)
        self.loss_weight = {k:v for k, v in zip(cfg.LOSS.TERM, cfg.LOSS.WEIGHTS)}
        self._build_model()

    def _build_model(self):
        self.s_img_feat_extractor = ResNetFeatureExtractor() # resnet18
        self.pcd_feat_extractor = PointNetfeat() 
        self.t_img_feat_extractor = ResNetFeatureExtractor() # resnet18
        self.feat_fusion = FusionRes() 
        self.align1 = AlignModule(64) # ResBlock lambda
        self.align2 = AlignModule(128) # ResBlock lambda
        self.align3 = AlignModule(256) # ResBlock lambda
        self.align4 = AlignModule(512) # ResBlock lambda
        if self.ind_feat: # switch for F2
            self.compress = nn.Sequential( # the linear layer in F2
                nn.Linear(2240, 1024),
                nn.BatchNorm1d(1024), nn.ReLU(inplace=True)
            )
        else:
            self.compress = nn.Sequential(
                nn.Linear(1984, 1024),
                nn.BatchNorm1d(1024), nn.ReLU(inplace=True)
            )

        self.rot_reg = nn.Sequential(
            nn.Linear(1024, 4)
        )

        self.z_reg = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1), nn.ReLU(),)
        self.amodal_reg = nn.Sequential(nn.Linear(1984, 400), nn.ReLU(), nn.Linear(400, 4)) # centric uv, amodal hw

    def forward(self, data):
        s_img = data['source_image']
        s_pcd = data['source_pcd']
        s_rot = data['source_rot']
        s_z = data['source_z']
        s_amodal = data['source_amodal']
        t_img = data['target_image']
        t_pcd = data['target_pcd']
        t_rot = data['target_rot']
        t_z = data['target_z']
        t_amodal = data['target_amodal']
        bs = s_img.shape[0]
        
        # ====== image feature extractor ====== #
        # source image + source feature extractor
        s_img_feat = self.s_img_feat_extractor(s_img)
        # target image + source feature extractor
        st_img_feat = self.s_img_feat_extractor(t_img) # the red data flow in the pipeline figure
        # target image + target feature extractor
        t_img_feat = self.t_img_feat_extractor(t_img)
        # ====== pcd feature extractor ====== # 
        s_pcd_feat, _, _ = self.pcd_feat_extractor(s_pcd)
        t_pcd_feat, _, _ = self.pcd_feat_extractor(t_pcd)

        # ====== align feature ======  #
        '''
        create linked space
        '''
        stc1 = self.align1(st_img_feat[0])
        stc2 = self.align2(st_img_feat[1])
        stc3 = self.align3(st_img_feat[2])
        stc4 = self.align4(st_img_feat[3])
        sc1 = self.align1(s_img_feat[0])
        sc2 = self.align2(s_img_feat[1])
        sc3 = self.align3(s_img_feat[2])
        sc4 = self.align4(s_img_feat[3])

        # fuse features
        s_pose, s_pose_in = self.feat_fusion(s_img_feat, [s_pcd_feat, None])
        ts_pose, ts_pose_in = self.feat_fusion([sc1, sc2, sc3, sc4, None], [s_pcd_feat, None])
        t_pose, t_pose_in = self.feat_fusion(t_img_feat, [t_pcd_feat, None])
        st_pose, st_pose_in = self.feat_fusion([stc1, stc2, stc3, stc4, None], [t_pcd_feat, None])
        if self.ind_feat:
            s_feat = torch.cat([s_pose, s_pose_in], dim=1)
            t_feat = torch.cat([t_pose, t_pose_in], dim=1)
            st_feat = torch.cat([st_pose, st_pose_in], dim=1)
            ts_feat = torch.cat([ts_pose, ts_pose_in], dim=1)
            s1_feat = torch.cat([s_pose, ts_pose_in], dim=1)
            t1_feat = torch.cat([t_pose, st_pose_in], dim=1)
            st1_feat = torch.cat([st_pose, t_pose_in], dim=1)
            ts1_feat = torch.cat([ts_pose, s_pose_in], dim=1)
        else:
            s_feat = s_pose
            t_feat = t_pose
            st_feat = st_pose
            ts_feat = ts_pose
            s1_feat = s_pose
            t1_feat = t_pose
            st1_feat = st_pose
            ts1_feat = ts_pose
        # ====== pose estimation on target domain ====== #
        # g_S^{f,f}
        rot_s = F.normalize(self.rot_reg(self.compress(s_feat)))
        z_s = self.z_reg(self.compress(s_feat))
        # g_T^{t,t}
        rot_t = F.normalize(self.rot_reg(self.compress(t_feat)))
        z_t = self.z_reg(self.compress(t_feat))
        if self.training:
            # g_T^{l,l}
            rot_st = F.normalize(self.rot_reg(self.compress(st_feat)))
            z_st = self.z_reg(self.compress(st_feat))
            # g_S^{l,l}
            rot_ts = F.normalize(self.rot_reg(self.compress(ts_feat)))
            z_ts = self.z_reg(self.compress(ts_feat))

            # g_S^{f,l}
            rot_s1 = F.normalize(self.rot_reg(self.compress(s1_feat)))
            z_s1 = self.z_reg(self.compress(s1_feat))
            # g_T^{t,l}
            rot_t1 = F.normalize(self.rot_reg(self.compress(t1_feat)))
            z_t1 = self.z_reg(self.compress(t1_feat))
            # g_T^{l,t}
            rot_st1 = F.normalize(self.rot_reg(self.compress(st1_feat)))
            z_st1 = self.z_reg(self.compress(st1_feat))
            # g_S^{l,f}
            rot_ts1 = F.normalize(self.rot_reg(self.compress(ts1_feat)))
            z_ts1 = self.z_reg(self.compress(ts1_feat))

            amodal_st = self.amodal_reg(st_pose)
            amodal_ts = self.amodal_reg(ts_pose)    

        ########## amodal bbox
        amodal_s = self.amodal_reg(s_pose)
        amodal_t = self.amodal_reg(t_pose)
        

        ########## loss
        rot_loss = nn.L1Loss()
        z_loss = nn.L1Loss()
        amodal_loss = nn.SmoothL1Loss()
        feat_loss = nn.L1Loss()
        outputs = {}

        
        if self.training:
            outputs['source_rot'] = (rot_loss(rot_s, s_rot) + rot_loss(rot_s1, s_rot)) if self.ind_feat else rot_loss(rot_s, s_rot)
            outputs['source_z'] = (z_loss(z_s, s_z) + z_loss(z_s1, s_z)) if self.ind_feat else z_loss(z_s, s_z)
            outputs['source_amodal'] = amodal_loss(amodal_s, s_amodal) 
            outputs['ts_rot'] = (rot_loss(rot_ts, s_rot) + rot_loss(rot_ts1, s_rot)) if self.ind_feat else rot_loss(rot_ts, s_rot)
            outputs['ts_z'] = (z_loss(z_ts, s_z) + z_loss(z_ts1, s_z)) if self.ind_feat else z_loss(z_ts, s_z)
            outputs['ts_amodal'] = amodal_loss(amodal_ts, s_amodal)
            outputs['feat_loss'] = 1.*(feat_loss(stc1, t_img_feat[0]) + \
            feat_loss(stc2, t_img_feat[1]) + \
            feat_loss(stc3, t_img_feat[2]) + \
            feat_loss(stc4, t_img_feat[3]) + \
            feat_loss(sc1, s_img_feat[0]) + \
            feat_loss(sc2, s_img_feat[1]) + \
            feat_loss(sc3, s_img_feat[2]) + \
            feat_loss(sc4, s_img_feat[3])
            )

            lamb = bs
            if lamb == 0.:
                lamb = 1.
            outputs['st_rot'] = (rot_loss(rot_st * data['target_rot_flag'], t_rot * data['target_rot_flag']) / lamb * bs \
            + rot_loss(rot_st1 * data['target_rot_flag'], t_rot * data['target_rot_flag']) / lamb * bs) if self.ind_feat \
            else rot_loss(rot_st * data['target_rot_flag'], t_rot * data['target_rot_flag']) / lamb * bs

            outputs['st_z'] = (z_loss(z_st * data['target_flag'], t_z * data['target_flag']) / lamb * bs \
            + z_loss(z_st1 * data['target_flag'], t_z * data['target_flag']) / lamb * bs) if self.ind_feat \
            else z_loss(z_st * data['target_flag'], t_z * data['target_flag']) / lamb * bs

            outputs['st_amodal'] = amodal_loss(amodal_st * data['target_amodal_flag'], t_amodal * data['target_amodal_flag']) / lamb * bs
            outputs['t_rot'] = (rot_loss(rot_t * data['target_rot_flag'], t_rot * data['target_rot_flag']) / lamb * bs \
            + rot_loss(rot_t1 * data['target_rot_flag'], t_rot * data['target_rot_flag']) / lamb * bs) if self.ind_feat \
            else rot_loss(rot_t * data['target_rot_flag'], t_rot * data['target_rot_flag']) / lamb * bs

            outputs['t_z'] = (z_loss(z_t * data['target_flag'], t_z * data['target_flag']) / lamb * bs \
            + z_loss(z_t1 * data['target_flag'], t_z * data['target_flag']) / lamb * bs) if self.ind_feat \
            else z_loss(z_t * data['target_flag'], t_z * data['target_flag']) / lamb * bs

            outputs['t_amodal'] = amodal_loss(amodal_t * data['target_amodal_flag'], t_amodal * data['target_amodal_flag']) / lamb * bs
            for k in outputs:
                if k in self.loss_term:
                    outputs[k] *= self.loss_weight[k]
        else:
            ######## prediction results
            outputs['source_rot'] = rot_s 
            outputs['source_z'] = z_s 
            outputs['source_amodal'] = amodal_s
            outputs['target_rot'] = rot_t
            outputs['target_z'] = z_t
            outputs['target_amodal'] = amodal_t
        return outputs
