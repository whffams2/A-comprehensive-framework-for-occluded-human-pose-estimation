# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.use_sa = False
# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.UP_SCALE = 1

_C.INSTNACEPaste = CN()
_C.INSTNACEPaste.ANN_FILE = '/opt/data/private/parts_bmp_filter_done/part_anns.json'
_C.INSTNACEPaste.ROOT_DIR = '/opt/data/private/parts_bmp_filter_done/'
_C.INSTNACEPaste.SCALE_RANGE = (0.7, 1.3)
_C.INSTNACEPaste.part_img_size = [192, 256]
_C.INSTNACEPaste.num_part = (1,)

_C.GCN = CN()
_C.GCN.SKELETON_GRAPH = 1
_C.GCN.Non_Local = True
_C.GCN.CHANNELS = 128
_C.GCN.NUM_LAYERS = 4
_C.GCN.P_DROPOUT = None
_C.GCN.LR = 1e-3
_C.GCN.EDGE = [[1, 2], [0, 3], [0, 4], [1, 5], [2, 4], [3, 6, 7, 11], [4, 5, 8, 12],
               [5, 9], [6, 10], [7], [8], [5, 12, 13], [6, 11, 14], [11, 15], [12, 16],
               [13], [14]]

_C.GCN.ext_1 = [[3, 4], [2, 5], [1, 6], [0, 6, 7, 11], [0, 5, 8, 12],  # head
                [1, 8, 9, 12, 13], [2, 7, 10, 11, 14], [3, 6, 11], [4, 5, 12],
                [5], [6], [3, 6, 7, 14, 15], [4, 5, 8, 13, 16], [5, 12], [6, 11],
                [11], [12]]

_C.GCN.ext_2 = [[5, 6], [4, 6, 7, 11], [3, 5, 8, 12], [2, 4, 8, 9, 12, 13],
                [1, 3, 7, 10, 11, 14], [0, 2, 6, 10, 14, 15], [0, 1, 5, 9, 13, 16],
                [1, 4, 8, 12, 13], [2, 3, 7, 11, 14], [3, 6, 11], [4, 5, 12],
                [1, 4, 5, 8, 9, 16], [2, 3, 6, 7, 10, 15], [3, 6, 7, 14],
                [4, 5, 8, 13], [5, 12], [6, 11]]

_C.GCN.ext_3 = [[5, 6, 7, 8, 11, 12], [6, 9, 8, 12, 13], [5, 7, 10, 11, 14],
                [4, 10, 11, 14, 15], [3, 9, 12, 13, 16], [0, 2, 13, 16],
                [0, 1, 15, 14], [0, 2, 6, 10, 11, 14, 15], [0, 1, 5, 9, 12, 13, 16],
                [1, 4, 8, 12, 13], [2, 3, 7, 11, 14], [0, 2, 10, 11, 14],
                [0, 1, 9, 12, 13], [1, 4, 5, 8, 9, 12, 16], [2, 3, 6, 7, 10, 11, 15],
                [3, 6, 7, 12], [4, 5, 8, 13]]

_C.GCN.Crowd_EDGE = [[2, 6, 13], [3, 7, 13], [0, 4], [1, 5],
                     [2], [3],
                     [0, 7, 8], [1, 6, 9], [6, 10], [7, 11],
                     [8], [9], [13], [0, 1, 12]]

_C.GCN.Crowd_ext_1 = [[1, 12, 4, 7, 8], [0, 12, 5, 6, 9], [6, 13], [7, 13],
                      [0], [1],
                      [1, 9, 2, 13, 10], [3, 13, 11, 0, 8], [0, 7], [1, 6],
                      [6], [7], [0, 1], [3, 7, 2, 6]]

_C.GCN.Crowd_ext_2 = [[3, 7, 1, 9, 10], [2, 6, 11, 0, 8], [1, 12, 7, 8], [0, 12, 6, 9], [6, 13], [7, 13],
                      [1, 3, 12, 13, 4, 11], [0, 2, 12, 13, 10, 5], [2, 13, 1, 9], [3, 13, 0, 8],
                      [0, 7], [1, 6], [2, 3, 6, 7], [4, 5, 8, 9]]

_C.GCN.Crowd_ext_3 = [[5, 13, 9, 6, 11], [4, 13, 7, 8, 10], [3, 7, 9, 10], [2, 6, 8, 11],
                      [1, 12, 7, 8], [0, 12, 6, 9],
                      [5, 3, 7, 13, 12, 0], [12, 13, 2, 4, 6, 1], [3, 13, 12, 11, 4, 1], [5, 10, 12, 0, 2, 13],
                      [9, 1, 13, 2], [8, 0, 13, 3], [5, 9, 8, 4], [10, 11]]

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.PROB_MASK_JOINTS = 0.0
_C.DATASET.PROB_PASTE = 0.0
_C.DATASET.USE_MASK_JOINTS = False
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.INTERVAL = 5
_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)