# -------------------------------------------------------------
# Modified from 'Point Transformer' 
# Reference: https://github.com/POSTECH-CVLab/point-transformer
# -------------------------------------------------------------
import sys
sys.dont_write_bytecode = True 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from util import anom_utils

random.seed(123)
np.random.seed(123)


##############################################################
def eval_open_measure(conf, seg_label, out_labels, mask=None):
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores  = - conf[np.logical_not(out_label)]
    out_scores = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any open pixels.")
        return 0.0, 0.0, 0.0
##############################################################


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    ##############################################################################
    if args.arch == 'pointtransformer_seg_repro' and args.open_eval != 'pointcam':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    elif args.arch == 'pointtransformer_seg_repro' and args.open_eval == 'pointcam':   
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro_pointcam as Model 
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

    if args.cutmix: 
        args.classes = args.classes + 1 
    ##############################################################################

    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name):
    data_path = os.path.join(args.data_root, data_name + '.npy')
    data = np.load(data_path)  # xyzrgbl, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test = 10
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    data_list = data_prepare()

    ######################## 
    seg_preds = np.array([])
    targets = np.array([])
    ########################  

    for idx, item in enumerate(data_list):
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
            pred, label = np.load(pred_save_path), np.load(label_save_path)
        else:
            coord, feat, label, idx_data = data_load(item)
            pred        = torch.zeros((label.size, args.classes)).cuda()
            #################################################
            pred_attent = torch.zeros((label.size)).cuda()
            coord_parts = torch.zeros((label.size, 3)).cuda()
            feat_parts  = torch.zeros((label.size, 3)).cuda()
            #################################################
            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            for i in range(idx_size):
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                idx_part = idx_data[i]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
                if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.voxel_max]
                        coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                else:
                    coord_part, feat_part = input_normalize(coord_part, feat_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)
            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                #####################
                with torch.no_grad():
                    if args.open_eval == "pointcam":
                        pred_part, attent1, attent2, attent3, w1, w2, w3 = model([coord_part, feat_part, offset_part])  # (n, k)
                        attent_part = w1*attent1 + w2*attent2 + w3*attent3
                    else: pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
                    
                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part

                if args.open_eval == "pointcam":
                    pred_attent[idx_part] += attent_part
                coord_parts[idx_part, :] += coord_part
                feat_parts[idx_part, :]  += feat_part
                ######################################
                logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))

            ###################################
            if args.open_eval == 'msp':
                seg_pred_msp    = F.softmax(pred, dim=-1)
                seg_pred_msp, _ = torch.max(seg_pred_msp, dim=-1)
                seg_preds = np.append(seg_preds, seg_pred_msp.cpu().data.numpy())
            elif args.open_eval == 'maxlogit':
                seg_pred_maxlogit, _ = torch.max(pred, dim=-1)
                seg_preds = np.append(seg_preds, seg_pred_maxlogit.cpu().data.numpy())
            elif args.open_eval == 'pointcam':
                seg_preds = np.append(seg_preds, (1-pred_attent).cpu().data.numpy())
            targets = np.append(targets, label)
            ###################################

            loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
            pred = pred.max(1)[1].data.cpu().numpy()

            ################################      
            if args.data_split == 's3dis_1':
                list_class_unknown = (10, )
            elif args.data_split == 's3dis_3':
                list_class_unknown = (7, 8, 10)

            auroc, aupr, fpr = eval_open_measure(np.array(seg_preds), np.array(targets), list_class_unknown, mask=None)
            print('Open-set result: AUROC/AUPR/FPR {:.4f}/{:.4f}/{:.4f}.'.format(auroc, aupr, fpr))
            #######################################################################################

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, batch_time=batch_time, accuracy=accuracy))
        pred_save.append(pred); label_save.append(label)
        np.save(pred_save_path, pred); np.save(label_save_path, label)

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)

    ################################
    if args.data_split == 's3dis_1':
        iou_class_tmp      = np.delete(iou_class, [10, 13], 0)
        accuracy_class_tmp = np.delete(accuracy_class, [10, 13], 0)
    elif args.data_split == 's3dis_3':
        iou_class_tmp      = np.delete(iou_class, [7, 8, 10, 13], 0)
        accuracy_class_tmp = np.delete(accuracy_class, [7, 8, 10, 13], 0)

    mIoU = np.mean(iou_class_tmp)
    mAcc = np.mean(accuracy_class_tmp)

    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    ################################

    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    # for i in range(args.classes):
    #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
