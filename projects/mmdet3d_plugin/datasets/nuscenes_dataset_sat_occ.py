# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_occ import NuScenesDatasetOccpancy

import torch



colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register_module()
class NuScenesDatasetSATOccpancy(NuScenesDatasetOccpancy):
    def __init__(self, *args, ifsat=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sat = ifsat

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        if self.sat:
            input_dict['sat_path'] = self.data_infos[index]['sat_path']

        # ── 关键修复：PrepareImageInputs 需要 results['curr'] ──
        input_dict['curr'] = self.data_infos[index]

        return input_dict
