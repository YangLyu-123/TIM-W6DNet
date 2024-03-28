import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys 
sys.path.insert(0, os.path.abspath("/media/yanglyu/DRIVING_1/PoseEstimation/codes/W6DNet")) # put your own path here
import argparse
import logging
import json
import time 
from tqdm import tqdm 
import numpy as np 
from contextlib import redirect_stdout
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset.apollo_style_dataset import ApolloStyleDataset
from dataset.utils import proj2Dto3D
from configs.config import get_cfg
from lib.w6dnet import *
from evaluation.eval_render import evaluate_rot, evaluate_trans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file', default='')
    parser.add_argument('--weights', type=str, help='.pth weight file', default='')
    parser.add_argument('--output', type=str, help='output path for saving results', default='')
    parser.add_argument("--mode", default='val')
    args = parser.parse_args()
    return args

def test(model, dataloader, cfg, output_dir, index=0, device='cuda:0'):
    gt_source_euler = []
    gt_source_z = []
    gt_source_amodal = []
    gt_target_euler = []
    gt_target_z = []
    gt_target_amodal = []
    pred_source_euler = []
    pred_source_z = []
    pred_source_amodal = []
    pred_target_euler = []
    pred_target_z = []
    pred_target_amodal = []
    source_fids = set()
    target_fids = set()
    source_results = {}
    target_results = {}

    source_feat_dir = os.path.join(output_dir, 'source_feat')
    target_feat_dir = os.path.join(output_dir, 'target_feat')
    os.makedirs(source_feat_dir, exist_ok=True)
    os.makedirs(target_feat_dir, exist_ok=True)

    for idx, data in enumerate(tqdm(dataloader)):
        source_fid = data['source_fid'][0] if data['source_fid'][0] not in source_fids else None
        target_fid = data['target_fid'][0] if data['target_fid'][0] not in target_fids else None
        
        if source_fid is not None:
            source_fids.add(source_fid)
            gt_source_euler.append(np.array(R.from_quat(data['source_rot']).as_euler('xyz', degrees=False)).reshape(3))
            gt_source_z.append(data["source_z"].item())
            gt_source_amodal.append([data['source_amodal'][0,0].item(), data['source_amodal'][0,1].item(), data['source_amodal'][0,2].item(), data['source_amodal'][0,3].item()])
            source_results[source_fid] = {}
            source_results[source_fid]['gt_rot'] = [gt_source_euler[-1][i] for i in range(3)]
            source_results[source_fid]['gt_z'] = gt_source_z[-1]
            source_results[source_fid]['gt_amodal'] = gt_source_amodal[-1]
            
        if target_fid is not None:
            target_fids.add(target_fid)
            gt_target_euler.append(np.array(R.from_quat(data['target_rot']).as_euler('xyz', degrees=False)).reshape(3))
            gt_target_z.append(data["target_z"].item())
            gt_target_amodal.append([data['target_amodal'][0,0].item(), data['target_amodal'][0,1].item(), data['target_amodal'][0,2].item(), data['target_amodal'][0,3].item()])
            target_results[target_fid] = {}
            target_results[target_fid]['gt_rot'] = [gt_target_euler[-1][i] for i in range(3)]
            target_results[target_fid]['gt_z'] = gt_target_z[-1]
            target_results[target_fid]['gt_amodal'] = gt_target_amodal[-1]
        
        data['source_image'] = data['source_image'].to(device)
        data['source_pcd'] = data['source_pcd'].to(device)
        data['source_rot'] = data['source_rot'].to(device)
        data['source_z'] = data['source_z'].to(device)
        data['source_amodal'] = data['source_amodal'].to(device)
        data['target_image'] = data['target_image'].to(device)
        data['target_pcd'] = data['target_pcd'].to(device)
        data['target_rot'] = data['target_rot'].to(device)
        data['target_z'] = data['target_z'].to(device)
        data['target_amodal'] = data['target_amodal'].to(device)

        ######## predictions
        predictions = model(data)
        if source_fid is not None:
            pred_source_euler.append(np.array(R.from_quat(predictions["source_rot"].detach().cpu().numpy().squeeze()).as_euler('xyz', degrees=False)).reshape(3))
            pred_source_z.append(predictions['source_z'].detach().cpu().numpy().squeeze())
            pred_source_amodal.append(predictions['source_amodal'].detach().cpu().numpy().squeeze())
            source_results[source_fid]['pred_rot'] = [pred_source_euler[-1][i] for i in range(3)]
            source_results[source_fid]['pred_z'] = pred_source_z[-1].item() 
            source_results[source_fid]['pred_amodal'] = [pred_source_amodal[-1][i].item() for i in range(4)]
        if target_fid is not None:
            pred_target_euler.append(np.array(R.from_quat(predictions["target_rot"].detach().cpu().numpy().squeeze()).as_euler('xyz', degrees=False)).reshape(3))
            pred_target_z.append(predictions['target_z'].detach().cpu().numpy().squeeze())
            pred_target_amodal.append(predictions['target_amodal'].detach().cpu().numpy().squeeze())
            target_results[target_fid]['pred_rot'] = [pred_target_euler[-1][i] for i in range(3)]
            target_results[target_fid]['pred_z'] = pred_target_z[-1].item() 
            target_results[target_fid]['pred_amodal'] = [pred_target_amodal[-1][i].item()  for i in range(4)]

    with open(os.path.join(output_dir, f'epoch{index}-source_results.json'), 'w+') as file:
        json.dump(source_results, file, indent=4)
    with open(os.path.join(output_dir, f'epoch{index}-target_results.json'), 'w+') as file:
        json.dump(target_results, file, indent=4)
    
def set_logger(log_file):
    logger = logging.getLogger()
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

def main(args):
    
    output = args.output
    os.makedirs(output, exist_ok=True)

    ######### cfg file
    config_file = args.config
    cfg = get_cfg()
    cfg.merge_from_file(config_file)    
    device = 'cuda:0'

    ######### set dataset
    mode = args.mode

    test_dataset = eval(cfg.DATASET.NAME)(cfg, mode=mode)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    weight_path = args.weights
    model = eval(cfg.MODEL.TYPE)(cfg)
    print("Processing {}...".format(weight_path))
    weights = torch.load(weight_path)
    model.load_state_dict(weights["state_dict"])
    model.to(device)
    model.eval()
    if 'epoch' in weights.keys():
        test(model, test_dataloader, cfg, output, weights['epoch'], device=device)
    else:
        test(model, test_dataloader, cfg, output, 1, device=device)
    
    

if __name__ == "__main__":
    # args
    args = get_args()
    main(args)