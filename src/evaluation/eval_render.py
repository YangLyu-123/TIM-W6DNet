import numpy as np
from navpy import angle2dcm as a2d
from scipy.linalg import logm, svd

def evaluate_rot(pred_euler, gt_euler, n=180, only_verbose=1):
    # https://github.com/ShapeNet/RenderForCNN/blob/master/view_estimation/compute_recall_precision_accuracy_3dview.m
    gt_R = a2d(gt_euler[:,0], gt_euler[:,1], gt_euler[:,2])
    pred_R = a2d(pred_euler[:,0], pred_euler[:,1], pred_euler[:,2])
    if len(gt_R.shape) == 2:
        gt_R = gt_R[None, :, :]
    if len(pred_R.shape) == 2:
        pred_R = pred_R[None, :, :]
    diff_R = np.matmul(pred_R.transpose(0, 2, 1), gt_R)
    # https://nl.mathworks.com/help/matlab/ref/norm.html#d123e955559 matlab --> numpy
    res = [np.max(svd(logm(diff_R[i,:,:].squeeze()))[1]) / np.sqrt(2) for i in range(diff_R.shape[0])]
    res = np.array(res)
    acc = np.sum(res < np.pi/180 * n) / res.shape[0]
    mederr = np.median(res)
    out_info = '[*] Rotation Evaluation \nAcc(pi/{}) = {:.2f}%, Mederr = {:.2f}(deg)\n'.format(180//n, acc * 100., mederr / np.pi * 180.)
    if only_verbose:
        return out_info
    else:
        return acc, mederr, out_info

def evaluate_trans(pred_trans, gt_trans, mode='XYZ', only_verbose=1):
    abs_diff = np.abs(gt_trans - pred_trans)
    
    if mode == 'XYZ':
        sq_diff = np.sqrt(np.sum(np.multiply(abs_diff, abs_diff), axis=1))
        gt_sq = np.sqrt(np.sum(np.multiply(gt_trans, gt_trans), axis=1))
        sq_rel_error = np.mean([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])])
        error_x = np.mean(abs_diff[:,0] / np.abs(gt_trans[:,0]))
        error_y = np.mean(abs_diff[:,1] / np.abs(gt_trans[:,1]))
        error_z = np.mean(abs_diff[:,2] / np.abs(gt_trans[:,2]))
        out_info = '[*] Translation Evaluation \nRelative L2 Error = {:.2f}%\n'.format(sq_rel_error * 100)
        out_info += f'[-] [eX, eY, eZ] (%) = [{error_x * 100}, {error_y * 100}, {error_z * 100}] \n'
        if only_verbose:
            return out_info
        else:
            return error_x, error_y, error_z, out_info
    else:
        sq_diff = np.sqrt(np.multiply(abs_diff, abs_diff))
        gt_sq = np.sqrt(np.multiply(gt_trans, gt_trans))
        sq_rel_error = np.mean([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])])
        out_info = '[*] Translation Evaluation \nRelative L2 Error = {:.2f}%\n'.format(sq_rel_error * 100)
        if only_verbose:
            return out_info
        else:
            return sq_rel_error * 100, out_info

