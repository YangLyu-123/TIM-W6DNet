import os 
import glob
import json
import re 
import pickle
import lmdb
from PIL import Image 
import random
import cv2 as cv
import numpy as np
import torch 
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R

from .apollo_car_model import car_name2id
from .utils import load_pcd, ratio_keep_resize, resized_centric, random_crop

random.seed(1)

class ApolloStyleDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super(ApolloStyleDataset).__init__()
        self.mode = mode
        ##### initialize dataset parameters
        self.initialize_dataset(cfg)
        ##### load annotations
        self.load_annotations()
        self.load_lmdb()
        ##### shuffle
        if 'train' in self.mode:
            self.shuffle()
        print('num apollo', len(self.apollo_indices), 'num style', len(self.style_indices))
        ##### load id info
        self.load_id_name()
        ##### augmentation
        self.augment()
        ##### load pcd 
        self.load_pcd()

    def initialize_dataset(self, cfg):
        self.resize = cfg.DATASET.RESIZE
        self.n_pcd = cfg.DATASET.N_PCD
        self.root = cfg.DATASET.ROOT
        self.use_template = cfg.DATASET.TEMPLATE_PCD
        self.apollo_label_set = set(cfg.DATASET.APOLLO_LABEL_SET)
        self.apollo_root = os.path.join(self.root, 'apollo')
        self.render_root = os.path.join(self.root, 'render')
        self.style_root = os.path.join(self.root, 'style')
        self.style_set = cfg.DATASET.SYN_SET
        self.apollo_K = np.array(
            [
                [2304.54786556982, 0, 1686.23787612802],
                [0.0, 2305.875668062, 1354.98486439791],
                [0.0, 0.0, 1.0]
            ]
        )
        self.render_K = np.array(
            [
                [1331.0226, 0, 960.0],
                [0.0, 1331.0226, 540.0],
                [0.0, 0.0, 1.0]
            ]
        ) ###### reference camera

    def load_apollo_annotations(self):
        self.apollo_img_root = self.apollo_root
        self.apollo_annotations = []
        if self.mode == 'train_small':
            for subset in self.apollo_label_set:
                with open(os.path.join(self.apollo_root, 'split_100', f'{subset}.json'), 'r') as file:
                    self.apollo_annotations.extend(json.load(file))
        elif self.mode == 'val':
            with open(os.path.join(self.apollo_root, f'{self.mode}.json'), 'r') as file:
                self.apollo_annotations = json.load(file)
        else:
            for subset in self.apollo_label_set:
                with open(os.path.join(self.apollo_root, f'train_part{subset}-5.json'), 'r') as file:
                    self.apollo_annotations.extend(json.load(file))
        self.load_apollo_label()

    def load_style_annotations(self):
        if 'val' in self.mode:
            mode = 'val' 
            anno_file = os.path.join(self.style_root, f'{mode}.json')
            with open(anno_file, 'r') as file:
                self.style_annotations = json.load(file)
        else:
            mode = self.mode
            if len(self.style_set) == 0:
                anno_file = os.path.join(self.style_root, f'train.json')
                with open(anno_file, 'r') as file:
                    self.style_annotations = json.load(file)

    def load_annotations(self):
        self.load_apollo_annotations()
        self.load_style_annotations() 
        self.apollo_indices = list(range(len(self.apollo_annotations)))
        self.style_indices = list(range(len(self.style_annotations)))

    def load_id_name(self):
        self.apollo_name2id = {name: car_name2id[name].id for name in car_name2id}
        self.apollo_id2name = {car_name2id[name].id: name for name in car_name2id}
        with open(os.path.join(self.render_root, 'label2model.json'), 'r') as file:
            self.render_id2name = json.load(file)
        with open(os.path.join(self.render_root, 'model2label.json'), 'r') as file:
            self.render_name2id = json.load(file)
        self.style_name2id = self.render_name2id
        self.style_id2name = self.render_id2name

    def load_apollo_label(self):
        if (not 'train' in self.mode) or len(self.apollo_label_set) == 0:
            self.apollo_label_idx = set()
            return
        self.apollo_label_idx = set()
        for subset in self.apollo_label_set:
            if isinstance(subset, int):
                with open(os.path.join(self.apollo_root, f'train_part{subset}-5.json'), 'r') as file:
                    data = json.load(file)
            else:
                with open(os.path.join(self.apollo_root, 'split_100', f'{subset}.json'), 'r') as file:
                    data = json.load(file)
            for item in data:
                self.apollo_label_idx.add(item['fid'])

    def load_lmdb(self):
        if self.mode == 'test' or self.mode == 'val':
            mode = 'val'
        else:
            mode = 'train'
        self.apollo_img_db_dict = lmdb.open(os.path.join(self.apollo_img_root, 'lmdb', f'{mode}.lmdb'), readonly=True, lock=False, readahead=False, meminit=False)
        self.apollo_meta_info_dict = pickle.load(open(os.path.join(self.apollo_img_root, 'lmdb', f'{mode}.lmdb', 'meta_info.pkl'), "rb"))
        if self.mode == 'test' or 'val' in self.mode:
            mode = 'val'
        else:
            mode = 'train'
        self.style_img_db_dict = lmdb.open(os.path.join(self.style_root, 'lmdb_padding0pix', f'{mode}.lmdb'), readonly=True, lock=False, readahead=False, meminit=False)
        self.style_meta_info_dict = pickle.load(open(os.path.join(self.style_root, 'lmdb_padding0pix', f'{mode}.lmdb', 'meta_info.pkl'), "rb"))
        
    def load_pcd(self):
        self.apollo_pcd = {}
        self.style_pcd = {}

        for idx in self.apollo_id2name:
            self.apollo_pcd[int(idx)] = load_pcd(os.path.join(self.apollo_root, 'pcd', f'{self.apollo_id2name[idx]}.pcd'))
        for idx in self.style_id2name:
            self.style_pcd[int(idx)] = load_pcd(os.path.join(self.style_root, 'pcd', f'{self.style_id2name[idx]}.pcd'))
        self.pcd_template = load_pcd('/media/yanglyu/YangLyu_Data5/YangLyu/PoseEstimation/dataset/cross/style/pcd/audi a7.001.pcd')# for testing the pcd template

    def load_image(self, db_reader, meta_info, key, img_type=np.uint8):
        with db_reader.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=img_type)
        key_index = list(meta_info['keys']).index(key)
        C,H,W = [int(s) for s in meta_info['resolution'][key_index].split('_')]
        image = img_flat.reshape(H,W,C)
        if img_type == np.uint16:
            image = np.array(image // 256).astype(np.uint8)
        return image

    def shuffle(self):
        random.shuffle(self.apollo_indices)
        random.shuffle(self.style_indices)

    def augment(self):
        if 'train' in self.mode:
            self.rgb_transform = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5),        
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                    ToTensorV2()
                ]) 
        else:
            self.rgb_transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                    ToTensorV2()
                ]
            )

    def __len__(self):
        if 'train' in self.mode:
            if len(self.style_annotations) > len(self.apollo_annotations):
                return len(self.style_annotations) 
            else:
                return len(self.apollo_annotations)
        else:
            return len(self.apollo_annotations)

    def __getitem__(self, index):
        if index == 0 and 'train' in self.mode:
            self.shuffle()
        apollo_index = self.apollo_indices[index % len(self.apollo_indices)]
        style_index = self.style_indices[index % len(self.style_indices)]

        apollo_annos = self.apollo_annotations[apollo_index] # target
        style_annos = self.style_annotations[style_index] # ts


        ########### load images
        a_key = f'{apollo_annos["fid"]}-{apollo_annos["car_id"]}-{apollo_annos["area"]}-{int(apollo_annos["bbox"][0])}'
        apollo_image = self.load_image(self.apollo_img_db_dict, self.apollo_meta_info_dict, a_key)
        sfid = style_annos['fid']
        s_key = f'{sfid}-{int(style_annos["centric"][0])}-{int(style_annos["centric"][1])}'
        style_image = self.load_image(self.style_img_db_dict, self.style_meta_info_dict, s_key)#

        ########### crop bbox
        style_bbox = style_annos['bbox'] # YXYX
        style_centric = style_annos['centric'] # UV
        style_bimg = style_image.copy()
        apollo_bbox = apollo_annos['bbox'] # XYXY
        apollo_bimg = apollo_image.copy()#
        apollo_centric = apollo_annos['centric']
        
        if 'train' == self.mode:
            ##### random crop
            style_bimg, sx1, sy1, sx2, sy2 = random_crop(style_bimg, int(style_bbox[1]), int(style_bbox[0]), int(style_bbox[3]), int(style_bbox[2]))
            apollo_bimg, ax1, ay1, ax2, ay2 = random_crop(apollo_bimg, int(apollo_bbox[0]), int(apollo_bbox[1]), int(apollo_bbox[2]), int(apollo_bbox[3]))
        else:
            sx1, sy1, sx2, sy2 = int(style_bbox[1]), int(style_bbox[0]), int(style_bbox[3]), int(style_bbox[2])
            ax1, ay1, ax2, ay2 = int(apollo_bbox[0]), int(apollo_bbox[1]), int(apollo_bbox[2]), int(apollo_bbox[3])
        ########### resize
        apollo_img, a_n_h, a_n_w, a_top, a_bottom, a_left, a_right = ratio_keep_resize(apollo_bimg, self.resize, self.resize)
        style_img, s_n_h, s_n_w, s_top, s_bottom, s_left, s_right = ratio_keep_resize(style_bimg, self.resize, self.resize)
        style_ncentric, style_amodal, style_ratio = resized_centric(style_centric, sx1, sy1, sx2, sy2, s_n_h, s_n_w, s_top, s_bottom, s_left, s_right, resize=self.resize)
        apollo_ncentric, apollo_amodal, apollo_ratio = resized_centric(apollo_centric, ax1, ay1, ax2, ay2, a_n_h, a_n_w, a_top, a_bottom, a_left, a_right, resize=self.resize)
        
        outputs = {}
        ###### image
        outputs['target_image'] = self.rgb_transform(image=apollo_img)['image']
        outputs['source_image'] = self.rgb_transform(image=style_img)['image']
        ### normal case
        outputs['target_pcd'] = torch.tensor(self.apollo_pcd[apollo_annos['car_id']], dtype=torch.float32)
        outputs['source_pcd'] = torch.tensor(self.style_pcd[style_annos['vid']], dtype=torch.float32)
        ###### z
        outputs['source_z'] = torch.tensor([style_annos['Trans'][2]  * self.apollo_K[0,0] / self.render_K[0,0] / style_ratio], dtype=torch.float32)
        ###### rotation
        outputs['source_rot'] = torch.tensor(R.from_euler('xyz', style_annos['euler_XYZ'], degrees=False).as_quat(), dtype=torch.float32)
        ###### amodal
        outputs['source_amodal'] = torch.tensor([style_ncentric[0]*1./self.resize, style_ncentric[1]*1./self.resize, np.log(style_amodal[0]*1./self.resize), np.log(style_amodal[1]*1./self.resize)], dtype=torch.float32)
        outputs['target_amodal'] = torch.tensor([apollo_ncentric[0]*1. / self.resize, apollo_ncentric[1]*1. / self.resize, np.log(apollo_amodal[0]*1./self.resize), np.log(apollo_amodal[1]*1./self.resize)], dtype=torch.float32)
        outputs['target_z'] = torch.tensor([apollo_annos['pose'][5]/apollo_ratio], dtype=torch.float32)
        if self.use_template:
            outputs['source_pcd'] = torch.tensor(self.pcd_template, dtype=torch.float32)
            outputs['target_pcd'] = torch.tensor(self.pcd_template, dtype=torch.float32)
        rot = np.array(R.from_euler('xyz', apollo_annos['pose'][:3], degrees=False).as_matrix())
        rot90 = np.array(
            [
                [1,0,0],
                [0,0,1],
                [0,-1,0]
            ]
        )
        rot = rot @ rot90 # calibrate the coordinates of synthetic and apollo
        outputs['target_rot'] = torch.tensor(np.array(R.from_matrix(rot).as_quat()), dtype=torch.float32)
        if apollo_annos['fid'] in self.apollo_label_idx or self.mode == 'val':
            outputs['target_flag'] = torch.tensor([1.], dtype=torch.float32)
            outputs['target_rot_flag'] = torch.tensor([1., 1., 1., 1.], dtype=torch.float32)
            outputs['target_amodal_flag'] = torch.tensor([1., 1., 1., 1.], dtype=torch.float32)
        else:
            outputs['target_flag'] = torch.tensor([0.], dtype=torch.float32)
            outputs['target_rot_flag'] = torch.tensor([0., 0., 0., 0.], dtype=torch.float32)
            outputs['target_amodal_flag'] = torch.tensor([0., 0., 0., 0.], dtype=torch.float32)
        outputs['target_z_flag'] = torch.tensor([1.], dtype=torch.float32)
        outputs['source_z_flag'] = torch.tensor([1.], dtype=torch.float32)
        if self.mode == 'val':
            outputs['target_vid'] = apollo_annos['car_id']
            outputs['source_vid'] = style_annos['vid']
            outputs['target_fid'] = a_key
            outputs['source_fid'] = s_key
            outputs['target_offset'] = torch.tensor([a_top, a_bottom, a_left, a_right], dtype=torch.float32)
            outputs['source_offset'] = torch.tensor([s_top, s_bottom, s_left, s_right], dtype=torch.float32)
            outputs['target_ratio'] = torch.tensor([apollo_ratio], dtype=torch.float32)
            outputs['source_ratio'] = torch.tensor([style_ratio], dtype=torch.float32)
            outputs['target_bbox'] = torch.tensor([apollo_bbox[0], apollo_bbox[1], apollo_bbox[2], apollo_bbox[3]], dtype=torch.float32)
            outputs['source_bbox'] = torch.tensor([style_bbox[1], style_bbox[0], style_bbox[3], style_bbox[2]], dtype=torch.float32)
        return outputs



