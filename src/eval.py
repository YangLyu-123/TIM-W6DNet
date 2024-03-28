import os 
import sys 
sys.path.insert(0, os.path.abspath("/media/yanglyu/DRIVING_1/PoseEstimation/codes/W6DNet"))  # put your own path here
import argparse
import numpy as np 
import cv2 as cv 
import json 
import glob 
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
from configs.config import get_cfg
from dataset.apollo_car_model import car_id2name
from dataset.utils import proj2Dto3D

from evaluation.eval_render import *


# FX: 2304.54786556982
# FY: 2305.875668062
# CX: 1686.23787612802
# CY: 1354.98486439791
HEIGHT = 2710
WIDTH = 3384

K = np.array([
    [2304.54786556982, 0, 1686.23787612802],
    [0.0, 2305.875668062, 1354.98486439791],
    [0.0, 0.0, 1.0]
])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", help='prediction results path', default='')
    parser.add_argument("--anno", help='ground truth file', default='./paper_results/val.json') # ground truth file
    parser.add_argument("--bbox_info", help='bbox information', default='./paper_results/apollo_style_apollo_val_bbox_info.json') # bbox information
    args = parser.parse_args()
    return args

def convert2apollo(args):
    output = args.pred_root
    os.makedirs(os.path.join(output, 'vis'), exist_ok=True)
    with open(args.bbox_info, 'r') as file:
        bbox_info = json.load(file)
    with open(glob.glob(os.path.join(output, '*target_results.json'))[0], 'r') as file:
        t_data = json.load(file)
    with open(os.path.join(args.anno), 'r') as file:
        gt = json.load(file)
    out_objs = []
    for gt_obj in gt:
        out_obj = {}
        for k in gt_obj:
            out_obj[k] = gt_obj[k]
        fid = f'{gt_obj["fid"]}-{gt_obj["car_id"]}-{gt_obj["area"]}-{int(gt_obj["bbox"][0])}'
        obj = t_data[fid]
        gt_euler = obj['gt_rot']
        gt_z = obj['gt_z']
        gt_amodal = obj['gt_amodal']
        pred_euler = obj['pred_rot']
        pred_z = obj['pred_z']
        pred_amodal = obj['pred_amodal']
        vid = fid.split('-')[1]
        frame = fid.split('-')[0]
        ######### rotation # coordinate conversion
        rot90 = np.array(
            [
                [1,0,0],
                [0,0,-1],
                [0,1,0]
            ]
        )
        gt_mat = np.array(R.from_euler('xyz', gt_euler, degrees=False).as_matrix())
        gt_mat = gt_mat @ rot90
        pred_mat = np.array(R.from_euler('xyz', pred_euler, degrees=False).as_matrix())
        pred_mat = pred_mat @ rot90
        ######### translation
        offset = bbox_info[fid]['target_offset']
        ratio = bbox_info[fid]['target_ratio']
        bbox = bbox_info[fid]['target_bbox']
        pred_z = pred_z * ratio 
        gt_z = gt_z * ratio
        top, left = offset[0], offset[2]
        #### amodal bbox
        gt_u, gt_v, gt_w, gt_h = gt_amodal
        gt_u = gt_u * 256
        gt_v = gt_v * 256
        gt_h = np.exp(gt_h) * 256 / (256 - offset[0] - offset[1]) * (bbox[3] - bbox[1])
        gt_w = np.exp(gt_w) * 256 / (256 - offset[2] - offset[3]) * (bbox[2] - bbox[0])
        pred_u, pred_v, pred_w, pred_h = pred_amodal
        pred_u = pred_u * 256
        pred_v = pred_v * 256
        pred_h = np.exp(pred_h) * 256 / (256 - offset[0] - offset[1]) * (bbox[3] - bbox[1])
        pred_w = np.exp(pred_w) * 256 / (256 - offset[2] - offset[3]) * (bbox[2] - bbox[0])
        if top > 0:
            gt_v -= top 
            pred_v -= top
        if left > 0:
            gt_u -= left 
            pred_u -= left
        gt_u, gt_v, pred_u, pred_v = gt_u / ratio, gt_v / ratio, pred_u / ratio, pred_v / ratio 
        gt_u = gt_u + bbox[0]
        gt_v = gt_v + bbox[1]
        pred_u = pred_u + bbox[0]
        pred_v = pred_v + bbox[1]
        pred_bbox = [np.max((pred_u - pred_w, 0)), np.max((pred_v - pred_h, 0)), np.min((pred_u + pred_w, WIDTH - 1)), np.min((pred_v + pred_h, HEIGHT - 1))]
        gt_bbox = [np.max((gt_u - gt_w, 0)), np.max((gt_v - gt_h, 0)), np.min((gt_u + gt_w, WIDTH - 1)), np.min((gt_v + gt_h, HEIGHT - 1))]
        ######### projection 3D --> 2D
        gt_trans = proj2Dto3D([gt_u, gt_v], gt_z, K)
        pred_trans = np.array(proj2Dto3D([pred_u, pred_v], pred_z, K)).reshape((1,3))
        pred_rot = np.array(R.from_matrix(pred_mat).as_euler('xyz', degrees=False)).reshape((1,3))
        out_obj['gt_pose'] = out_obj['pose']
        out_obj['pose'] = [pred_rot[0,0], pred_rot[0,1], pred_rot[0,2], pred_trans[0,0], pred_trans[0,1], pred_trans[0,2]]
        out_obj['centric'] = [int(pred_u), int(pred_v)]
        out_obj['pred_bbox'] = [pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]]
        out_obj['gt_bbox'] = [gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3]]
        out_objs.append(out_obj)
    return out_objs

def evaluation(data):
    gt_trans = []
    pred_trans = []
    gt_euler = []
    pred_euler = []

    for obj in data:
        gt_euler.append([obj['gt_pose'][0], obj['gt_pose'][1], obj['gt_pose'][2]])
        pred_euler.append([obj['pose'][0], obj['pose'][1], obj['pose'][2]])
        gt_trans.append([obj['gt_pose'][3], obj['gt_pose'][4], obj['gt_pose'][5]])
        pred_trans.append([obj['pose'][3], obj['pose'][4], obj['pose'][5]])

    gt_euler = np.array(gt_euler)
    pred_euler = np.array(pred_euler)
    gt_trans = np.array(gt_trans)
    pred_trans = np.array(pred_trans)

    rot_info = evaluate_rot(pred_euler, gt_euler, n=30)
    trans_info = evaluate_trans(pred_trans, gt_trans)
    print(rot_info)
    print(trans_info)
    return rot_info, trans_info

def main(args):
    out_objs = convert2apollo(args)
    rot_info, trans_info = evaluation(out_objs)
    output = args.pred_root
    with open(os.path.join(output, 'final_pred_pbbox.json'), 'w') as file:
        json.dump(out_objs, file, indent=4)
    with open(os.path.join(output, 'evaluation.txt'), 'w') as file:
        file.write(rot_info)
        file.write(trans_info)

if __name__ == "__main__":
    args = get_args()
    main(args)
