# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Linhao Xu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.loss import JointsMSELoss
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.inference import get_gcn_max_preds
from core.inference import get_gcn_preds

logger = logging.getLogger(__name__)


def train(config, train_loader, model, gcn, criterion, optimizer,
          gcn_optimizer, epoch, output_dir, tb_log_dir, writer_dict):
    # def train(config, train_loader, model, criterion, optimizer,
    #           epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    criterion_heatmap = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        device = input.device
        outputs = model(input)
        gts = meta['joints'][:, :, :2].to(device)
        # joints_ori = meta['joints_ori'].to(device)
        center = meta['center'].to(device)
        scale = meta['scale'].to(device)
        bbox = meta['bbox']
        B, C, H, W = outputs.shape
        preds, maxvals = get_gcn_max_preds(outputs)
        dts = get_gcn_preds(preds, center, scale, W, H)  # aug_ori_dts
        gts = get_gcn_preds((gts / 2).clone(), center, scale, W, H)  # aug_ori_gts
        # gts = get_gcn_preds((gts / 4).clone(), center, scale, W, H)  # aug_ori_gts
        # hm = hm_normalize(dts.clone(), H, W)  # 64 17 2
        dts, gts = normalize(dts, bbox, gts)
        # gcn 回归一个offset
        results_off = gcn(dts)
        results_preds = results_off + dts
        gts = gts[:, :, :2].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        if isinstance(outputs, list):
            loss_heatmap = criterion_heatmap(outputs[0], target, target_weight)
            for outputs in outputs[1:]:
                loss_heatmap += criterion_heatmap(outputs, target, target_weight)
        else:
            output = outputs
            loss_heatmap = criterion_heatmap(output, target, target_weight)

        if isinstance(results_preds, list):
            loss = criterion(results_preds[0], gts, target_weight)
            for results_preds in results_preds[1:]:
                loss += criterion(results_preds, gts, target_weight)
        else:
            output = results_preds
            loss = criterion(output, gts, target_weight)

        loss = loss_heatmap + loss
        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step

        optimizer.zero_grad()
        gcn_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gcn_optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)


def validate(config, val_loader, val_dataset, model, gcn, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    # def validate(config, val_loader, val_dataset, model, criterion, output_dir,
    #              tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    criterion_heatmap = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            bbox = meta['bbox']
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            dts, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            dts = torch.from_numpy(dts).to(device=output.device)
            dts, _ = normalize(dts, bbox, is_train=False)

            output_result = gcn(dts)
            output_result = output_result + dts
            output_result = inverse_normalize_only(output_result, bbox)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion_heatmap(output, target, target_weight)
            # loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # preds, maxvals = get_final_preds(
            #     config, output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = output_result[:, :, 0:2].cpu().numpy()
            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def gcn_train(config, train_loader, model, gcn, criterion, criterion_heatmap, criterion_sem, optimizer,
              gcn_optimizer, epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    losses = AverageMeter()
    multi_losses = AverageMeter()
    sem_losses = AverageMeter()
    gcn_losses = AverageMeter()
    acc = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, sem_labels, meta) in enumerate(
            train_loader):
        # for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        device = input.device
        outputs, multi_scores, sem_scores = model(input)
        # outputs = model(input)
        gts = meta['joints'][:, :, :2].to(device)
        # joints_ori = meta['joints_ori'].to(device)
        center = meta['center'].to(device)
        scale = meta['scale'].to(device)
        bbox = meta['bbox']
        B, C, H, W = outputs.shape
        preds, maxvals = get_gcn_max_preds(outputs)
        dts = get_gcn_preds(preds, center, scale, W, H)  # aug_ori_dts

        gts = get_gcn_preds((gts / 2).clone(), center, scale, W, H)  # aug_ori_gts
        hm = hm_normalize(dts.clone(), H, W)  # 64 17 2
        dts, gts = normalize(dts, bbox, gts)
        # gcn 回归一个offset
        
        results_off = gcn(dts, hm, outputs)
        results_preds = results_off + dts
        gts = gts[:, :, :2].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        sem_labels = sem_labels.cuda(non_blocking=True)
        if isinstance(outputs, list):
            loss_heatmap = criterion_heatmap(outputs[0], target, target_weight)
            for outputs in outputs[1:]:
                loss_heatmap += criterion_heatmap(outputs, target, target_weight)
        else:
            output = outputs
            loss_heatmap = criterion_heatmap(output, target, target_weight)

        if isinstance(multi_scores, list):
            multi_loss = criterion_heatmap(multi_scores[0], all_ins_target, all_ins_target_weight)
            for output in multi_scores[1:]:
                multi_loss += criterion_heatmap(output, all_ins_target, all_ins_target_weight)
        else:
            output = multi_scores
            multi_loss = criterion_heatmap(output, all_ins_target, all_ins_target_weight)

        if isinstance(results_preds, list):
            gcn_loss = criterion(results_preds[0], gts, target_weight)
            for results_preds in results_preds[1:]:
                gcn_loss += criterion(results_preds, gts, target_weight)
        else:
            output = results_preds
            gcn_loss = criterion(output, gts, target_weight) * 0.01

        sem_loss = criterion_sem(sem_scores, sem_labels) * 0.00001

        total_loss = loss_heatmap + multi_loss + sem_loss + gcn_loss

        # loss = loss_heatmap + loss_multi_heatmap + loss * 0.01
        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step

        optimizer.zero_grad()
        gcn_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        gcn_optimizer.step()

        # measure accuracy and record loss
        losses.update(loss_heatmap.item(), input.size(0))
        multi_losses.update(multi_loss.item(), input.size(0))
        sem_losses.update(sem_loss.item(), input.size(0))
        gcn_losses.update(gcn_loss.item(), input.size(0))
        total_losses.update(total_loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'total_Loss {total_loss.val:.5f} ({total_loss.avg:.5f})\t' \
                  'loss {Loss.val:.5f} ({Loss.avg:.5f})\t' \
                  'multi_Loss {multi_loss.val:.5f} ({multi_loss.avg:.5f})\t' \
                  'sem_Loss {sem_loss.val:.5f} ({sem_loss.avg:.5f})\t' \
                  'gcn_Loss {gcn_loss.val:.5f} ({gcn_loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, total_loss=total_losses, Loss=losses, multi_loss=multi_losses,
                sem_loss=sem_losses, gcn_loss=gcn_losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', total_losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def gcn_validate(config, val_loader, val_dataset, model, gcn, criterion, output_dir,
                 tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
    criterion_heatmap = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, sem_labels, meta) in enumerate(
                val_loader):
            
            # compute output
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            bbox = meta['bbox']
            outputs, _, _ = model(input)
            # outputs, _ = model(input)
            B, C, H, W = outputs.shape
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, _, _ = model(input_flipped)
                # outputs_flipped, _ = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            dts, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            dts = torch.from_numpy(dts).to(device=output.device)
            dts, _ = normalize(dts, bbox, is_train=False)

            hm = hm_normalize(dts.clone(), H, W)  # 64 17 2
            # gcn 回归一个offset
            results_off = gcn(dts, hm, outputs)
            output_result = results_off + dts
            output_result = inverse_normalize_only(output_result, bbox)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion_heatmap(output, target, target_weight)
            # loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # preds, maxvals = get_final_preds(
            #     config, output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = output_result[:, :, 0:2].cpu().numpy()
            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred * 4, output,
                #                   prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def hm_normalize(x, h, w):
    x[:, :, 0] /= w
    x[:, :, 1] /= h
    x -= 0.5
    x *= 2
    return x


def normalize(dts, bbox, gts=None, is_train=True):
    num_joints = dts.shape[1]
    device = dts.device
    dts[:, :, 0] = dts[:, :, 0] - bbox[0].to(device).unsqueeze(-1).repeat(1, num_joints)
    dts[:, :, 1] = dts[:, :, 1] - bbox[1].to(device).unsqueeze(-1).repeat(1, num_joints)
    if is_train:
        gts = gts.to(device)
        gts[:, :, 0] = gts[:, :, 0] - bbox[0].to(device).unsqueeze(-1).repeat(1, num_joints)
        gts[:, :, 1] = gts[:, :, 1] - bbox[1].to(device).unsqueeze(-1).repeat(1, num_joints)

    for bz in range(dts.shape[0]):
        w = bbox[2]
        h = bbox[3]
        w = w.to(device)
        h = h.to(device)
        dts[bz, :, :2] = normalize_screen_coordinates(dts[bz, :, :2], w[bz], h[bz])
        if is_train:
            gts[bz, :, :2] = normalize_screen_coordinates(gts[bz, :, :2], w[bz], h[bz])

    return dts, gts


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    # Normalize
    X[:, 0] = X[:, 0] / float(w) - 0.5
    X[:, 1] = X[:, 1] / float(h) - 0.5
    return X * 2


def inverse_normalize_only(dts, bbox):
    num_joints = dts.shape[1]

    for bz in range(dts.shape[0]):
        w = bbox[2]
        h = bbox[3]
        dts[bz, :, :2] = inverse_normalize(dts[bz, :, :2], w[bz], h[bz])
    dts[:, :, 0] = dts[:, :, 0] + bbox[0].to(dts.device).unsqueeze(-1).repeat(1, num_joints)
    dts[:, :, 1] = dts[:, :, 1] + bbox[1].to(dts.device).unsqueeze(-1).repeat(1, num_joints)
    return dts


def inverse_normalize(Y, w, h):
    assert Y.shape[-1] == 2

    Y /= 2.
    Y += 0.5
    Y[:, 0] = Y[:, 0] * float(w)
    Y[:, 1] = Y[:, 1] * float(h)
    return Y


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
