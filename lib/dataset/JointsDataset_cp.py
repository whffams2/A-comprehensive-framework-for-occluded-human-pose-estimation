# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Linhao Xu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# import mmcv
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

logger = logging.getLogger(__name__)


class JointsDatasetCP(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.to_float32 = False
        self.color_type = 'color'
        self.channel_order = 'rgb'
        # self.file_client_args = dict(backend='disk')
        # self.file_client = mmcv.FileClient(**self.file_client_args)

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.up_scale_factor = cfg.MODEL.UP_SCALE
        self.use_mask_joints = cfg.DATASET.USE_MASK_JOINTS
        self.use_sa = cfg.use_sa
        self.prob_mask_joints = cfg.DATASET.PROB_MASK_JOINTS
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self, ):
        return len(self.db)

    def mask_joint(self, image, joints, MASK_JOINT_NUM=3):
        ## N,J,2 joints
        N = joints.shape[0]
        height, width, _ = image.shape

        size = np.random.randint(10, 20, (N, 2))

        x0 = np.array(joints[:, 0], dtype=int) - size[:, 0]
        y0 = np.array(joints[:, 1], dtype=int) - size[:, 1]

        x1 = np.array(joints[:, 0], dtype=int) + size[:, 0]
        y1 = np.array(joints[:, 1], dtype=int) + size[:, 1]

        np.clip(x0, 0, width)
        np.clip(x1, 0, width)
        np.clip(y0, 0, height)
        np.clip(y1, 0, height)
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        ind = np.random.choice(N, MASK_JOINT_NUM)
        for j in ind:
            image[y0[j]:y1[j], x0[j]:x1[j]] = 0

        return image
    
    # def _read_image(self, path):
    #     img_bytes = self.file_client.get(path)
    #     img = mmcv.imfrombytes(
    #         img_bytes, flag=self.color_type, channel_order=self.channel_order)
    #     if img is None:
    #         raise ValueError(f'Fail to read {path}')
    #     if self.to_float32:
    #         img = img.astype(np.float32)
    #     return img
    #
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        # data_numpy = self._read_image(image_file)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        if 'interference' in db_rec.keys():
            interference_joints = db_rec['interference']
            interference_joints_vis = db_rec['interference_vis']
        else:
            interference_joints = [joints]
            interference_joints_vis = [joints_vis]
        c = db_rec['center']
        s = db_rec['scale']
        bbox = db_rec['bbox']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                for i in range(len(interference_joints)):
                    interference_joints[i], interference_joints_vis[i] = fliplr_joints(
                        interference_joints[i], interference_joints_vis[i], data_numpy.shape[1], self.flip_pairs)

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        if self.is_train:
            if self.use_mask_joints:
                if np.random.rand() < self.prob_mask_joints:
                    input = self.mask_joint(input, joints, MASK_JOINT_NUM=2)
            if self.use_sa:
                if np.random.randn() < self.paste_prob:
                    # if np.random.randn() < 1:
                    t_type_idx = random.randint(0, len(self.instance_proveider.candidate_parts) - 1)
                    t_type = self.instance_proveider.candidate_parts[t_type_idx]
                    part_ann = random.sample(self.instance_proveider.part_anns[t_type], 1)[0]
                    path = os.path.join(self.instance_proveider.part_root_dir, t_type, part_ann['file_name'])
                    part_img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    part_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB)
                    scale_factor = random.uniform(*self.instance_proveider.scale_range)
                    # part_joints = np.array(part_ann['keypoints']).reshape(-1, 3)
                    # part_img, part_joints = self.instance_proveider. \
                    #     adjust_scale_by_diagnal(part_img, part_joints,
                    #                             bbox,
                    #                             self.instance_proveider.part_scale[t_type],
                    #                             scale_factor)
                    part_mask = (cv2.cvtColor(part_img, cv2.COLOR_RGB2GRAY) > 0).astype(np.float32)
                    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    part_mask = cv2.erode(part_mask, erode_kernel, 1)
                    part_mask = cv2.GaussianBlur(part_mask, (3, 3), 0)
                    part_mask = part_mask[:, :, np.newaxis]
                    Hp, Wp = part_mask.shape[:2]
                    center = np.array([Hp // 2, Wp // 2])
                    scale = self.image_size / 200
                    trans = get_affine_transform(center, scale, 0.0, self.image_size)
                    part_img = cv2.warpAffine(part_img, trans,
                                              (int(self.image_size[0]), int(self.image_size[1])),
                                              flags=cv2.INTER_LINEAR)
                    part_mask = cv2.warpAffine(part_mask, trans,
                                               (int(self.image_size[0]), int(self.image_size[1])),
                                               flags=cv2.INTER_LINEAR)
                    mask = np.zeros_like(input)
                    mask[:, :, 0] = part_mask
                    mask[:, :, 1] = part_mask
                    mask[:, :, 2] = part_mask
                    input = input * (1 - mask) + part_img * mask
                    # cv2.imwrite('aug_img'+part_ann['file_name'][:-4] + '.jpg', input)

        if self.transform:
            input = self.transform(input)

        target, target_weight = self.generate_target(joints, joints_vis)

        inter_target = np.zeros_like(target)
        inter_target_weight = np.zeros_like(target_weight)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        for i in range(len(interference_joints)):
            inter_joints = interference_joints[i].copy()
            inter_joints_vis = interference_joints_vis[i].copy()
            for j in range(self.num_joints):
                if inter_joints_vis[j, 0] > 0.0:
                    inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
            _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)

            inter_target = np.maximum(inter_target, _inter_target)
            inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)

        all_ins_target = np.maximum(0.5 * inter_target, target)
        all_ins_target_weight = np.maximum(inter_target_weight, target_weight)
        if self.up_scale_factor > 1:
            self.heatmap_size *= 2
            self.sigma += 1
            target, target_weight = self.generate_target(joints, joints_vis)
            inter_target = np.zeros_like(target)
            inter_target_weight = np.zeros_like(target_weight)
            for i in range(len(interference_joints)):
                inter_joints = interference_joints[i].copy()
                inter_joints_vis = interference_joints_vis[i].copy()
                for j in range(self.num_joints):
                    if inter_joints_vis[j, 0] > 0.0:
                        inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
                _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)
                inter_target = np.maximum(inter_target, _inter_target)
                inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)
            all_ins_target = np.maximum(0.5 * inter_target, target)
            all_ins_target_weight = np.maximum(inter_target_weight, target_weight)
            sem_labels = (target_weight > 0).astype(np.float32)  # 17
            sem_labels = np.reshape(sem_labels, (-1,))  # 17,
            self.heatmap_size = self.heatmap_size // 2
            self.sigma -= 1

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'bbox': bbox,
            'rotation': r,
            'score': score,
        }

        # return input, target, target_weight, joints, c, s, bbox, meta
        return input, target, target_weight, all_ins_target, all_ins_target_weight, sem_labels, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
