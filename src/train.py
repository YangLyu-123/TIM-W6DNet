import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys 
sys.path.insert(0, os.path.abspath("/media/yanglyu/DRIVING_1/PoseEstimation/codes/W6DNet"))  # put your own path here
import argparse
import logging
import time 
import json
from tqdm import tqdm 
import numpy as np 
from contextlib import redirect_stdout
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset.apollo_style_dataset import ApolloStyleDataset
from configs.config import get_cfg
from lib.w6dnet import *
from evaluation.eval_render import evaluate_rot, evaluate_trans

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/CROSS/PAPER/ALL_SYN/CROSS_APOLLO_STYLE_FEAT_a0.20_s1.0.yaml', help='config file')
    args = parser.parse_args()
    return args

def set_logger(log_file):
    logger = logging.getLogger('w6dnet')
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

def train(model, optimizer, dataloader, writer, logger, device, idx):
    pbar = tqdm(total=len(dataloader))
    n = len(dataloader)
    for i, data in enumerate(dataloader):
        postfix = OrderedDict()
        postfix["Epoch"] = idx
        for key in data:
            data[key] = data[key].to(device)
        optimizer.zero_grad()
        loss_dict = model(data)
        out_loss_str = '--- Train ---'
        for k in loss_dict:
            postfix[k] = loss_dict[k].item()
            out_loss_str += " {}: {} ".format(k, loss_dict[k].item())
        loss = sum(loss_dict.values())
        postfix["total_loss"] = loss.item()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(postfix)
        pbar.update(1)
        for k in loss_dict:
            writer.add_scalar("{}".format(k), loss_dict[k].item(), idx * n + i)
            writer.add_scalar("total loss", loss.item(), idx * n + i)
    pbar.close()
               
def val_dual(model, dataloader, writer, output_dir, index, device='cuda:0'):
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
        
    source_acc, source_mederr, source_rot_info = evaluate_rot(np.array(pred_source_euler), np.array(gt_source_euler), n=30, only_verbose=0)
    source_z_err, source_z_info = evaluate_trans(np.array(pred_source_z), np.array(gt_source_z), mode='X', only_verbose=0)
    target_acc, target_mederr, target_rot_info = evaluate_rot(np.array(pred_target_euler), np.array(gt_target_euler), n=30, only_verbose=0)
    target_z_err, target_z_info = evaluate_trans(np.array(pred_target_z), np.array(gt_target_z), mode='X', only_verbose=0)
    out_info = '===== Source {} vehicles, Target {} vehicles \n'.format(len(gt_source_euler), len(gt_target_euler))
    out_info += '====================== Source Domain performance =====================\n'
    out_info += source_rot_info
    out_info += source_z_info
    out_info += '====================== Target Domain performance =====================\n'
    out_info += target_rot_info
    out_info += target_z_info
    print(out_info)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'epoch{index}-evaluation.txt'), 'w+') as file:
        file.write(out_info)
    with open(os.path.join(output_dir, f'epoch{index}-source_results.json'), 'w+') as file:
        json.dump(source_results, file, indent=4)
    with open(os.path.join(output_dir, f'epoch{index}-target_results.json'), 'w+') as file:
        json.dump(target_results, file, indent=4)
    writer.add_scalar("source rot acc(pi/6)", source_acc, index)
    writer.add_scalar("source rot mederr", source_mederr, index)
    writer.add_scalar("target rot acc(pi/6)", target_acc, index)
    writer.add_scalar("target rot mederr", target_mederr, index)
    writer.add_scalar("source z error", source_z_err, index)
    writer.add_scalar("target z error", target_z_err, index)

def main(args):
    ############ cfg file
    cfg = get_cfg()
    cfg.merge_from_file(args.config) 

    ########### cuda 
    device = cfg.DEVICE

    ######### timestamp
    now = time.localtime()
    timestamp = "{}-{}-{}-{}-{}-{}".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    ######### output
    output = cfg.OUTPUT
    output = os.path.join(output, "{}-{}".format(cfg.MODEL.TYPE, timestamp))
    os.makedirs(output, exist_ok=True)

    ######### log config
    config_log = os.path.join(output, "config.yaml")
    #### write config file
    with open(config_log, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())
    ######### model log
    model_log = os.path.join(output, "log.log")
    logger = set_logger(model_log)

    ######### set dataset
    train_dataset = eval(cfg.DATASET.NAME)(cfg, mode=cfg.DATASET.MODE)
    val_dataset = eval(cfg.DATASET.NAME)(cfg, mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAINER.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=4)

    ######### set model
    print("Initiating model {}.....".format(cfg.MODEL.TYPE))
    model = eval(cfg.MODEL.TYPE)(cfg)
    model.to(device)
    if 'W6DNet' in cfg.MODEL.TYPE:
        val_func = val_dual    

    if cfg.SOLVER.OPTIM == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.0001)
    elif cfg.SOLVER.OPTIM == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAINER.LR, momentum=0.9, weight_decay=0.0001)
    
    if cfg.SOLVER.LR_SCHEDULE == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAINER.EPOCH, eta_min=cfg.TRAINER.ETA_MIN)
    elif cfg.SOLVER.LR_SCHEDULE == "multi-lr":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAINER.LR_DECAY_STEP, gamma=cfg.TRAINER.LR_DECAY)

    if os.path.isfile(cfg.MODEL.WEIGHTS):
        print(f"Loading weights from {cfg.MODEL.WEIGHTS}")
        weights = torch.load(cfg.MODEL.WEIGHTS)
        model.load_state_dict(weights["state_dict"])
        start_epoch = 0
    else:
        start_epoch = 0

    ######## SummaryWriter
    train_writer = SummaryWriter(os.path.join(output, 'summary', 'train'))
    val_writer = SummaryWriter(os.path.join(output, 'summary', 'val'))
    n_epoch = cfg.TRAINER.EPOCH
    
    for idx in range(start_epoch, n_epoch):
        model.train()
        train(model, optimizer, train_dataloader, train_writer, logger, device, idx)
        state = {
            'epoch': idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }
        torch.save(state, os.path.join(output, "checkpoint.pth"))
        if 'W6DNet' in cfg.MODEL.TYPE and idx % 2 == 0:
            model.eval()
            val_func(model, val_dataloader, val_writer, os.path.join(output, 'checkpoint'), idx)
        for param_group in optimizer.param_groups:
                logger.info("epoch {}/{} learning rate = {}".format(idx, n_epoch, param_group['lr']))
        lr_scheduler.step()
        if idx % 5 == 0 or (n_epoch - idx) <= 25:
            state = {
                        'epoch': n_epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        }
            torch.save(state, os.path.join(output, "epoch{:03d}.pth".format(idx)))



if __name__ == '__main__':
    args = get_args()
    main(args)
