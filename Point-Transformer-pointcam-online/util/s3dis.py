# -------------------------------------------------------------
# Modified from 'Point Transformer' 
# Reference: https://github.com/POSTECH-CVLab/point-transformer
# -------------------------------------------------------------
import os

import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare


################################################################
def unknown_aug_deform(points, labels, select_ratio, NEW_LABEL):
    coords = points[:, :3]
    seed = coords[np.random.randint(0, points.shape[0]), :]
    dist = np.linalg.norm(coords - np.tile(np.expand_dims(seed, axis=0), (points.shape[0], 1)), axis=-1)

    select_ratio = np.random.uniform(low=0.0, high=select_ratio)
    # select_idx = np.argsort(dist)[:int(points.shape[0] * select_ratio)]
    select_idx = np.argpartition(-dist, -int(points.shape[0] * select_ratio))[-int(points.shape[0] * select_ratio):]

    labels[select_idx] = NEW_LABEL
    select_coords = coords[select_idx, :]

    # add rotation
    center = np.mean(select_coords, axis=0)
    unknown_coords = select_coords - center
    unknown_coords = np.matmul(unknown_coords, np.random.randn(3, 3))
    unknown_coords = unknown_coords + center

    # add translation
    max_value = np.max(coords, axis=0)
    min_value = np.min(coords, axis=0)
    x_pos = np.random.uniform(low=min_value[0], high=max_value[0], size=(1,))
    y_pos = np.random.uniform(low=min_value[1], high=max_value[1], size=(1,))
    z_pos = np.random.uniform(low=min_value[2], high=max_value[2], size=(1,))
    rand_center = np.concatenate((x_pos, y_pos, z_pos))
    unknown_coords = unknown_coords - (center - rand_center)

    # update new coords
    coords[select_idx, :] = unknown_coords
    points = np.concatenate((coords, points[:, 3:]), axis=-1)
    return points, labels
################################################################


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1, 
                 data_split=None, cutmix=None, select_ratio=None):
        super().__init__()
        ####################################
        if data_split == 's3dis_1':
            self.list_class_unknown = [10]
            self.NEW_LABEL = 13
        elif data_split == 's3dis_3':
            self.list_class_unknown = [7, 8, 10]
            self.NEW_LABEL = 13
        else: self.list_class_unknown = [13]

        if cutmix:
            self.cutmix = cutmix
            self.select_ratio = select_ratio
        ####################################

        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        ##################################
        if self.split == 'train':
            index_unknown = np.where(np.isin(label, self.list_class_unknown))
            label = np.delete(label, index_unknown, 0)
            coord = np.delete(coord, index_unknown, 0)
            feat  = np.delete(feat, index_unknown, 0)

        if self.split == 'train' and self.cutmix:
            coord = coord.numpy()
            label = label.numpy()
            feat  = feat.numpy()

            coord_old = coord
            coord, label = unknown_aug_deform(coord, label, self.select_ratio, self.NEW_LABEL)  
            feat = feat*(1 + (coord-coord_old)/(coord_old+1e-6))

            coord = torch.from_numpy(coord)
            label = torch.from_numpy(label)
            feat  = torch.from_numpy(feat)
        ##################################

        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
