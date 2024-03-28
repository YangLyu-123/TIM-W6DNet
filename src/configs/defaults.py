# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

##### device setting
_C.DEVICE = "cuda:0"

##### model setting
_C.MODEL = CN()
_C.MODEL.TYPE = ''
_C.MODEL.CLASSIFIER = False
_C.MODEL.WEIGHTS = ""
_C.MODEL.ADD_FEAT_TRANS = False
_C.MODEL.ROT_ADD_PCD = False
_C.MODEL.CENTRIC_Z_MULTIPLE_IMG_FEAT = False
_C.MODEL.CONTOUR = False
_C.MODEL.PCD_DISCRIMINATOR = True # different version of pcd discriminator
_C.MODEL.MONO_CHANNEL = False
_C.MODEL.ROTATION_MODE = 'QUATERNION'
_C.MODEL.REFINE = False
_C.MODEL.IND_FEAT = True

##### feature extractor
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "resnet50"
_C.BACKBONE.PRETRAINED = True
_C.BACKBONE.N_CLASSES = 79

##### camera
_C.CAMERA = CN()
_C.CAMERA.FX = 1331.0226
_C.CAMERA.FY = 1331.0226
_C.CAMERA.CX = 960.0
_C.CAMERA.CY = 540.0
_C.CAMERA.WIDTH = 1920
_C.CAMERA.HEIGHT = 1080

##### dataset
_C.DATASET = CN()
_C.DATASET.MODE = "train"
_C.DATASET.NAME = 'RenderDataset'
_C.DATASET.ROOT = "/media/yanglyu/YangLyu_Data4/PoseEstimation/dataset"
_C.DATASET.N_PCD = 1000
_C.DATASET.SET = ('city-CameraRenderSubStreet.001',)
_C.DATASET.SCENE = 'city'
_C.DATASET.AUGMENT = True
_C.DATASET.CAMERA_NAME = 'CameraRenderSubStreet.001'
_C.DATASET.RESIZE = 256
_C.DATASET.TEMPLATE_PCD = False
_C.DATASET.CROP = False
_C.DATASET.VALID_PCD = False
_C.DATASET.PADDING = False
_C.DATASET.APOLLO_LABEL = False
_C.DATASET.APOLLO_SET = ()
_C.DATASET.APOLLO_LABEL_SET = (0, )
_C.DATASET.SOURCE_STYLE_ALIGN = False
_C.DATASET.SYN_SET = ()
_C.DATASET.APOLLO_PSEUDO_SET = ()
_C.DATASET.BCS_VIEW = ''

##### output
_C.OUTPUT = ""

############ trainer
_C.TRAINER = CN()
_C.TRAINER.EPOCH = 10
_C.TRAINER.LR = 0.001
# save model per n_iter
_C.TRAINER.N_ITER = 5000
_C.TRAINER.BATCH_SIZE = 4
_C.TRAINER.LR_DECAY = 0.5
_C.TRAINER.LR_DECAY_STEP = (5000, 10000,)
_C.TRAINER.LOSS_WEIGHT_DECAY = 1000
_C.TRAINER.WARM_UP_STEP = 0
_C.TRAINER.MULTI_GPUS = False
_C.TRAINER.ETA_MIN = 1e-5

############ Loss
_C.LOSS = CN()
_C.LOSS.TYPE = ("pose", "cls", "mask", "render")
_C.LOSS.ADAPTIVE = False

_C.LOSS.TERM = ('source_rot', 'source_z', 'source_amodal', 'ts_rot', 'ts_z', 'ts_amodal', 'feat_loss', 'st_rot', 'st_z', 'st_amodal', 't_rot', 't_z', 't_amodal',)
_C.LOSS.WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, )

########## solver
_C.SOLVER = CN()
_C.SOLVER.OPTIM = "SGD"
_C.SOLVER.LR_SCHEDULE = "cosine"