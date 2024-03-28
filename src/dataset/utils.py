import open3d as o3d
import numpy as np
import torch
import array
import cv2 as cv
from scipy.spatial.transform import Rotation as R

def load_pcd(file):
    pcd = o3d.io.read_point_cloud(file)
    xyz = np.asarray(pcd.points).T 
    return xyz

def iou(bbox1, bbox2):###### XYXY
    inter_x1 = np.max((bbox1[0], bbox2[0]))
    inter_x2 = np.min((bbox1[2], bbox2[2]))
    inter_y1 = np.max((bbox1[1], bbox2[1]))
    inter_y2 = np.min((bbox1[3], bbox2[3]))
    intersection = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
    union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection
    return intersection / union

def ratio_keep_resize(bboxed_img, new_h, new_w): # only support new_h == new_w
    h, w, c = bboxed_img.shape
    assert w != 0
    assert h != 0
    ratio = h/w
    if ratio > 1:
        n_h = new_h
        n_w = int(new_h / ratio)
        top = 0
        bottom = 0
        left = (new_w - n_w) // 2
        right = (new_w - n_w - left)
    else:
        n_w = new_w
        n_h = int(new_w * ratio)
        top = (new_h - n_h) // 2
        bottom = (new_h - n_h - top)
        left = 0
        right = 0
    new_img = cv.resize(bboxed_img, (n_w, n_h))
    new_img = cv.copyMakeBorder(new_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, [0,0,0])
    if c == 1:
        new_img = new_img[:,:,np.newaxis]
    return new_img, n_h, n_w, top, bottom, left, right

def proj2Dto3D(pts2d, zs, k):
    fx = k[0,0]
    fy = k[1,1]
    cx = k[0,2]
    cy = k[1,2]
    X = (pts2d[0] - cx) * zs / fx
    Y = (pts2d[1] - cy) * zs / fy
    return np.array([X, Y, zs])

def quaternion_product(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)

def load_apollo_obj(filename_obj, normalization=False, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32))
    vertices[:, 1] = -vertices[:, 1]

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)) - 2
    return vertices, faces

def load_render_obj(filename_obj):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1
    return vertices, faces

def vis_obj(verts, faces, img, mask_color=(0,0,255)):
    for f in faces:
        coord = np.array([verts[:2, int(f[0])], verts[:2, int(f[1])], verts[:2, int(f[2])]], dtype=np.int32)
        cv.fillPoly(img, np.int32([coord]), mask_color)

def proj3Dto2D(X,K,RT):
    if X.shape[0] != 3:
        verts1 = X.T
    else:
        verts1 = X
    
    P = np.ones((verts1.shape[0] + 1, verts1.shape[1]))
    P[:3, :] = verts1[:3, :]
    
    coor = np.dot(K, np.dot(RT, P))

    coor[0,:] = coor[0,:] / coor[2,:]
    coor[1,:] = coor[1,:] / coor[2,:]
    return coor

def get_valid_pcd(K, ori_pcd, RT, bbox, n_pcd=1024):
    
    if ori_pcd.shape[0] != 3:
        tmp_pcd = ori_pcd.T
    else:
        tmp_pcd = ori_pcd # 3xn
    P = np.ones((tmp_pcd.shape[0] + 1, tmp_pcd.shape[1]))
    P[:3,:] = tmp_pcd[:3,:]
    coord = np.dot(K, np.dot(RT, P))
    coord[0,:] /= coord[2,:]
    coord[1,:] /= coord[2,:]
    idx_w = [i for i in range(tmp_pcd.shape[1]) if coord[0,i] >= bbox[0] and coord[0,i] <= bbox[2]]
    idx_wh = [i for i in idx_w if coord[1,i] >= bbox[1] and coord[1,i] <= bbox[3]]
    valid_code = np.zeros((n_pcd))
    valid_code[idx_wh] = 1
    selected_pcd = [np.array(tmp_pcd[:,i]).reshape(3) for i in idx_wh]
    if n_pcd - len(selected_pcd) > 0:
        duplicate_idx = np.random.randint(0, high=len(selected_pcd), size=(n_pcd-len(selected_pcd)))
    else: 
        duplicate_idx = None
    if duplicate_idx is None:
        return ori_pcd, valid_code
    dupli_pcd = [selected_pcd[i] for i in duplicate_idx]

    selected_pcd.extend(dupli_pcd)
    return np.array(selected_pcd).T, valid_code

def random_crop(img, bb_x1, bb_y1, bb_x2, bb_y2):
    ######## random crop for a bounded image
    image = img.copy()
    if (bb_y2 - bb_y1) > 300 and (bb_x2 - bb_x1) > 300:
        crop_ratio = np.random.uniform(0, 0.4)
    else:
        crop_ratio = np.random.uniform(0, 0.1)
    ##### left or right
    if_left = np.random.choice(2, 1, p=[0.5, 0.5])
    crop_width = (bb_x2 - bb_x1) * crop_ratio
    if if_left:
        bb_x1 += crop_width
        image = image[:, int(crop_width):, :]
    else:
        bb_x2 -= crop_width
        image = image[:, :int(bb_x2-bb_x1), :]
    
    ####### top or bottom
    if_top = np.random.choice(2, 1, p=[0.5, 0.5])
    crop_height = (bb_y2 - bb_y1) * crop_ratio
    if if_top:
        bb_y1 += crop_height
        image = image[int(crop_height):, :, :]
    else:
        bb_y2 -= crop_height
        image = image[:int(bb_y2 - bb_y1), :, :]
    return image, bb_x1, bb_y1, bb_x2, bb_y2

def resized_centric(centric, x1, y1, x2, y2, new_h, new_w, top=0, bottom=0, left=0, right=0, resize=256):
    r_cx = centric[0] - x1 
    r_cy = centric[1] - y1
    a_w = np.max([centric[0] - x1, x2 - centric[0]])
    a_h = np.max([centric[1] - y1, y2 - centric[1]])
    new_rcx = r_cx*1. / (x2-x1) * new_w 
    new_rcy = r_cy*1. / (y2-y1) * new_h
    if new_h > new_w:
        new_rcx += left 
        ratio = resize * 1. / (y2-y1)
    else:
        new_rcy += top
        ratio = resize * 1. / (x2-x1)
    return [new_rcx, new_rcy], [a_w*1./(x2-x1)*new_w, a_h*1./(y2-y1)*new_h], ratio

def get_transformed_pcd(pcd, quater, z):
    rot = R.from_quat(quater).as_matrix()
    trans_pcd = rot @ pcd
    pcd[2,:] += z
    return pcd

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]
    return int(cmin), int(rmin), int(cmax), int(rmax)