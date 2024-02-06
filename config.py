from yacs.config import CfgNode as CN
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union


_C = CN()

_C.Dist = False
_C.Local_rank = 0


_C.BASIC = CN()
_C.BASIC.Commit_Info = "Baseline"
_C.BASIC.Early_stop = 70
_C.BASIC.Epoch_dis = 15
_C.BASIC.Epochs = 300
_C.BASIC.DEBUG = False
_C.BASIC.Finetune = False
_C.BASIC.Lr_decay = 15
_C.BASIC.Num_gpus = "0"
_C.BASIC.Resume = False  # '/home/jovyan/projects/code/ocsr-xwj/output_runs/MARKUSH/UNet_VAE/finetune0.5-V4-latent64-maaV2_zero/07_04-14_00/Model/checkpoint.pth'
_C.BASIC.Seed = 14207
_C.BASIC.Use_wandb = True
_C.BASIC.Warmup_epoch = 20
_C.BASIC.no_trans_epoch = 15
_C.BASIC.view_mode = 2  # 0 for single view, and input fast view, 1 for single view, and input slow view, 2 for single view and input both slow view and fast view, 3 for dual view and input from both fast view and slow view from two views
_C.BASIC.OVA = False


_C.DATA = CN()
_C.DATA.Train = CN()
_C.DATA.Train.Class = "mTICI_Dual"
_C.DATA.Train.DataPara = CN()
_C.DATA.Train.DataPara.name = "mTICI_Dual_LMDB"
_C.DATA.Train.DataPara.state = "train"
_C.DATA.Train.DataPara.json_file_dir = "/ai/mnt/code/DSFNet_MTICI/Data/ReNamedAll.json"
# _C.DATA.Train.DataPara.fold = "FOLD1"
_C.DATA.Train.DataPara.fast_time_size = 8
_C.DATA.Train.DataPara.crop = (0.1, 0.1, 0.2, 0.1)
_C.DATA.Train.DataPara.visual_size = 256
_C.DATA.Train.DataPara.fuse01 = True
_C.DATA.Train.DataPara.binary = False
_C.DATA.Train.LoaderPara = CN()
_C.DATA.Train.LoaderPara.batch_size = 24
_C.DATA.Train.LoaderPara.num_workers = 8

_C.DATA.Val = CN()
_C.DATA.Val.Class = _C.DATA.Train.Class
_C.DATA.Val.DataPara = CN()
_C.DATA.Val.DataPara.name = _C.DATA.Train.DataPara.name
_C.DATA.Val.DataPara.state = "val"
# _C.DATA.Val.DataPara.fold = _C.DATA.Train.DataPara.fold
_C.DATA.Val.DataPara.json_file_dir = _C.DATA.Train.DataPara.json_file_dir
_C.DATA.Val.DataPara.fast_time_size = _C.DATA.Train.DataPara.fast_time_size
_C.DATA.Val.DataPara.crop = -1
_C.DATA.Val.DataPara.visual_size = _C.DATA.Train.DataPara.visual_size
_C.DATA.Val.DataPara.fuse01 = _C.DATA.Train.DataPara.fuse01
_C.DATA.Val.DataPara.binary = _C.DATA.Train.DataPara.binary
_C.DATA.Val.LoaderPara = CN()
_C.DATA.Val.LoaderPara.batch_size = _C.DATA.Train.LoaderPara.batch_size * 2
_C.DATA.Val.LoaderPara.num_workers = _C.DATA.Train.LoaderPara.num_workers * 2

if "Triple" in _C.DATA.Train.DataPara.json_file_dir:
    _C.DATA.Test = CN()
    _C.DATA.Test.Class = _C.DATA.Train.Class
    _C.DATA.Test.DataPara = CN()
    _C.DATA.Test.DataPara.name = _C.DATA.Train.DataPara.name
    _C.DATA.Test.DataPara.state = "test"
    _C.DATA.Test.DataPara.json_file_dir = _C.DATA.Train.DataPara.json_file_dir
    _C.DATA.Test.DataPara.fast_time_size = _C.DATA.Train.DataPara.fast_time_size
    _C.DATA.Test.DataPara.crop = -1
    _C.DATA.Test.DataPara.visual_size = _C.DATA.Train.DataPara.visual_size
    _C.DATA.Test.DataPara.fuse01 = _C.DATA.Train.DataPara.fuse01
    _C.DATA.Test.DataPara.binary = _C.DATA.Train.DataPara.binary
    _C.DATA.Test.LoaderPara = CN()
    _C.DATA.Test.LoaderPara.batch_size = 8
    _C.DATA.Test.LoaderPara.num_workers = 8
else:
    _C.DATA.Test = None

if _C.DATA.Train.Class == "mTICI_Single":
    sin = "SINGLE_VIEW"
else:
    sin = "DUAL_VIEW"

if (
    "All" in _C.DATA.Train.DataPara.json_file_dir
    and "All" in _C.DATA.Val.DataPara.json_file_dir
):
    sin = sin + "-ALL"
    if "Triple" in _C.DATA.Train.DataPara.json_file_dir:
        sin = sin + "-Tri"
if _C.DATA.Train.DataPara.binary:
    fu = "Binary"
    num_classes = 2
else:
    if _C.DATA.Train.DataPara.fuse01:
        fu = "fuse01"
        num_classes = 4
    else:
        fu = ""
        num_classes = 5


_C.OPT = CN()
_C.OPT.Name = "AdamW"
_C.OPT.Trans_scaler = 10
_C.OPT.Para = CN()
_C.OPT.Para.lr = 0.0003  # 0.0003
_C.OPT.Para.weight_decay = 0.01

_C.SCHEDULER = CN()
_C.SCHEDULER.Name = "CosineAnnealingWarmRestarts"
_C.SCHEDULER.Para = CN()
_C.SCHEDULER.Para.T_0 = 35
_C.SCHEDULER.Para.T_mult = 2
_C.SCHEDULER.Para.eta_min = 1.0e-6

# _C.SCHEDULER = CN()
# _C.SCHEDULER.Name = "Poly"
# _C.SCHEDULER.Para = CN()
# _C.SCHEDULER.Para.max_peoch = _C.BASIC.Epochs - _C.BASIC.Warmup_epoch
# _C.SCHEDULER.Para.initial_lr = _C.OPT.Para.lr


# _C.LOSS = CN()
# _C.LOSS.Name = "FocalLabelSmooth_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [1, 0.8],
#     [1, 0.8],
#     [1, 0.8],
#     [0.5, 0.4],
#     [0.5, 0.4],
#     [0.25, 0.2],
#     [0.25, 0.2],
#     [0.125, 0.1],
#     [0.125, 0.1],
# ]
# _C.LOSS.Para.smoothing = 0.4

# _C.LOSS = CN()
# _C.LOSS.Name = "FocalLabelSmoothOHEM_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [1, 0.8, 1.4],
#     [1, 0.8, 1.4],
#     [1, 0.8, 1.4],
#     [0.5, 0.4, 0.7],
#     [0.5, 0.4, 0.7],
#     [0.25, 0.2, 0.35],
#     [0.25, 0.2, 0.35],
#     [0.125, 0.1, 0.2],
#     [0.125, 0.1, 0.2],
# ]

# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.keep_rate = 0.55

# _C.LOSS = CN()
# _C.LOSS.Name = "FocalLabelSmoothSeasaw_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [1, 0.8, 1.4],
#     [1, 0.8, 1.4],
#     [1, 0.8, 1.4],
#     [0.5, 0.4, 0.7],
#     [0.5, 0.4, 0.7],
#     [0.25, 0.2, 0.35],
#     [0.25, 0.2, 0.35],
#     [0.125, 0.1, 0.2],
#     [0.125, 0.1, 0.2],
# ]
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.keep_rate = 0.55
# _C.LOSS.Para.num_classes = 4 if _C.DATA.Train.DataPara.fuse01 else 5
# _C.LOSS.Para.p = 0.8
# _C.LOSS.Para.q = 2.0
# _C.LOSS.Para.eps = 1e-2

_C.LOSS = CN()
_C.LOSS.Name = "LabelSmoothSeasaw_MISO"
_C.LOSS.Para = CN()
_C.LOSS.Para.balance = [
    [0.8, 1.2],
    [0.8, 1.2],
    [0.8, 1.2],
    [0.4, 0.6],
    [0.4, 0.6],
    [0.2, 0.3],
    [0.2, 0.3],
    [0.1, 0.15],
    [0.1, 0.15],
]
_C.LOSS.Para.smoothing = 0.4
_C.LOSS.Para.keep_rate = 0.55
_C.LOSS.Para.num_classes = num_classes
_C.LOSS.Para.p = 0.8  # 0.8
_C.LOSS.Para.q = 1.0  # 2
_C.LOSS.Para.eps = 1e-2

# _C.LOSS.Para.balance = [
#     [1, 0.8, 1.4],
#     [1, 0.8, 1.4],
#     [1, 0.8, 1.4],
#     [0.5, 0.4, 0.7],
#     [0.5, 0.4, 0.7],
#     [0.25, 0.2, 0.35],
#     [0.25, 0.2, 0.35],
#     [0.125, 0.1, 0.2],
#     [0.125, 0.1, 0.2],
# ]

# _C.LOSS = CN()
# _C.LOSS.Name = "FocalLabelSmoothOVA_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [1, 0.8],
#     [1, 0.8],
#     [1, 0.8],
#     [0.5, 0.4],
#     [0.5, 0.4],
#     [0.25, 0.2],
#     [0.25, 0.2],
#     [0.125, 0.1],
#     [0.125, 0.1],
# ]
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.num_classes = 4 if _C.DATA.Train.DataPara.fuse01 else 5
# _C.BASIC.OVA = True


# _C.LOSS = CN()
# _C.LOSS.Name = "FocalLabelSmoothOVA_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [1, 0.8, 1.25],
#     [1, 0.8, 1.25],
#     [1, 0.8, 1.25],
#     [0.5, 0.4, 0.625],
#     [0.5, 0.4, 0.625],
#     [0.25, 0.2, 0.625],
#     [0.25, 0.2, 0.625],
#     [0.125, 0.1, 0.31],
#     [0.125, 0.1, 0.31],
# ]
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.num_classes = 4 if _C.DATA.Train.DataPara.fuse01 else 5


# _C.LOSS = CN()
# _C.LOSS.Name = "FocalLabelSmoothSupclu_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [1, 0.8, 0.6],
#     [1, 0.8, 0.6],
#     [1, 0.8, 0.6],
#     [0.5, 0.4, 0.3],
#     [0.5, 0.4, 0.3],
#     [0.25, 0.2, 0.15],
#     [0.25, 0.2, 0.15],
#     [0.125, 0.1, 0.075],
#     [0.125, 0.1, 0.075],
# ]
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.temperature = 0.07
# _C.LOSS.Para.contrast_mode = "all"
# _C.LOSS.Para.base_temperature = 0.07

# balance=[
#             [1, 0.8, 1.25],
#             [1, 0.8, 1.25],
#             [1, 0.8, 1.25],
#             [0.5, 0.4, 0.625],
#             [0.5, 0.4, 0.625],
#             [0.25, 0.2, 0.625],
#             [0.25, 0.2, 0.625],
#             [0.125, 0.1, 0.31],
#             [0.125, 0.1, 0.31],
#         ]


# _C.LOSS = CN()
# _C.LOSS.Name = "FocalSupClu_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[3, 0.75, 0.8], [2.5, 0.75, 0.8], [2.5, 0.75, 0.8]]
# _C.LOSS.Para.temperature = 0.07
# _C.LOSS.Para.contrast_mode = "all"
# _C.LOSS.Para.base_temperature = 0.07

# _C.LOSS = CN()
# _C.LOSS.Name = "FocalOHEM_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]]
# _C.LOSS.Para.keep_rate = 0.8

# _C.LOSS = CN()
# _C.LOSS.Name = "GRWOHEM_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]]
# _C.LOSS.Para.keep_rate = 0.8
# _C.LOSS.Para.fuse01 = _C.DATA.Train.DataPara.fuse01
# _C.LOSS.Para.fold = _C.DATA.Train.DataPara.fold


# _C.LOSS = CN()
# _C.LOSS.Name = "GRWFOCAL_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]]
# _C.LOSS.Para.fuse01 = _C.DATA.Train.DataPara.fuse01
# _C.LOSS.Para.fold = _C.DATA.Train.DataPara.fold


# _C.LOSS = CN()
# _C.LOSS.Name = "OHEMLabelSmooth_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]]
# _C.LOSS.Para.keep_rate = 0.8
# _C.LOSS.Para.smoothing = 0.4


# _C.LOSS = CN()
# _C.LOSS.Name = "GRWLabelSmooth_MISO"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]]
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.fuse01 = _C.DATA.Train.DataPara.fuse01
# _C.LOSS.Para.fold = _C.DATA.Train.DataPara.fold

# _C.LOSS = CN()
# _C.LOSS.Name = "DistanceFocalLabelsoomth"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [[0.8, 1, 0.8], [0.8, 1, 0.8], [0.8, 1, 0.8], [0.8, 1, 0.8]]
# _C.LOSS.Para.loss_mode = "mse"
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.fuse01 = _C.DATA.Train.DataPara.fuse01


# _C.MODEL = CN()
# _C.MODEL.Name = "SLOW_FAST_NEW"
# _C.MODEL.Para = CN()
# _C.MODEL.Para.model_depth = 18
# _C.MODEL.Para.stage_drop = [-0.1, -0.2, -0.15, -0.1]
# _C.MODEL.Para.model_num_class = 4 if _C.DATA.Train.DataPara.fuse01 else 5
# _C.MODEL.Para.hourglass = [-1]
# _C.MODEL.Para.pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/SLOWFAST_8x8_R50-Kinect.pyth"


# _C.MODEL = CN()
# _C.MODEL.Name = "R3D"
# _C.MODEL.Para = CN()
# _C.MODEL.Para.model_depth = 50
# _C.MODEL.Para.model_num_class = 4 if _C.DATA.Train.DataPara.fuse01 else 5
# _C.MODEL.Para.pretrained = "/home/user/skip/code/DSFNet_MTICI/Src/SLOW_8x8_R50.pyth"

# _C.MODEL = CN()
# _C.MODEL.Name = "X3D"
# _C.MODEL.Para = CN()
# _C.MODEL.Para.input_clip_length = 16
# _C.MODEL.Para.input_crop_size = 512
# _C.MODEL.Para.use_marc = True
# _C.MODEL.Para.model_num_class = 4 if _C.DATA.Train.DataPara.fuse01 else 5
# _C.MODEL.Para.pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/X3D_M-Kinect.pyth"


# ff
_C.MODEL = CN()
_C.MODEL.Name = "DVCNet"
_C.MODEL.Para = CN()
_C.MODEL.Para.input_clip_length = _C.DATA.Train.DataPara.fast_time_size
_C.MODEL.Para.input_crop_size = _C.DATA.Train.DataPara.visual_size
_C.MODEL.Para.use_marc = True
_C.MODEL.Para.model_num_class = num_classes
_C.MODEL.Para.use_fusion = "CVFMTrans"  # [ADD,CONCAT,CVFM,CVFMTrans]
_C.MODEL.Para.mlp_dropout_rate = 0
_C.MODEL.Para.num_heads = 8
_C.MODEL.Para.expand_dim = 8
_C.MODEL.Para.deep_super = [False, True, False, True]
_C.MODEL.Para.OVA = _C.BASIC.OVA
_C.MODEL.Para.cor_pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/X3D_M-Kinect.pyth"
_C.MODEL.Para.sag_pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/X3D_M-Kinect.pyth"
_C.MODEL.Para.fuse_pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/X3D_M-Kinect.pyth"

# _C.MODEL = CN()
# _C.MODEL.Name = "DVCNet_R2Plus1D"
# _C.MODEL.Para = CN()
# _C.MODEL.Para.input_clip_length = _C.DATA.Train.DataPara.fast_time_size
# _C.MODEL.Para.input_crop_size = _C.DATA.Train.DataPara.visual_size
# _C.MODEL.Para.use_marc = True
# _C.MODEL.Para.model_num_class = num_classes
# _C.MODEL.Para.use_fusion = "CVFMTrans"  # [ADD,CONCAT,CVFM,CVFMTrans]
# _C.MODEL.Para.mlp_dropout_rate = 0
# _C.MODEL.Para.num_heads = 8
# _C.MODEL.Para.expand_dim = 8
# _C.MODEL.Para.deep_super = [False, True, False, True]
# _C.MODEL.Para.OVA = _C.BASIC.OVA
# _C.MODEL.Para.cor_pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/R2PLUS1D_16x4_R50.pyth"
# _C.MODEL.Para.sag_pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/R2PLUS1D_16x4_R50.pyth"
# _C.MODEL.Para.fuse_pretrained = "/ai/mnt/code/DSFNet_MTICI/Src/R2PLUS1D_16x4_R50.pyth"


_C.METHOD = CN()
_C.METHOD.Desc = (
    "%s-%s-T%02d#V%03d-RENAMED/COR_Kinect-SAG_Kinect-%s_OptScaler20-%s_EroDilK3-noclip-DS13-init0914-fixdataset-fixmlp-fixloss-fixdroplast-fixpretrained-try1"
    % (
        sin,
        fu,
        _C.DATA.Train.DataPara.fast_time_size,
        _C.DATA.Train.DataPara.visual_size,
        _C.MODEL.Para.use_fusion,
        _C.LOSS.Name,
    )
)
_C.METHOD.Detail_Desc = "oversample-lmdb-fixpre-Pre_withbeforetrain-4 sptial temoral position embedding(4STPE)"
_C.METHOD.Name = _C.MODEL.Name
