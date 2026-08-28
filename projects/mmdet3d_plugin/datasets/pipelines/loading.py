# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import random

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate
import torchvision.transforms as transforms

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = False # ← 修复：PIL 已经是 RGB，不需要转换
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

def mmlabNormalize_sat(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True # ← 修复：PIL 已经是 RGB，不需要转换
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

@PIPELINES.register_module()
class PrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            load_point_label=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.load_point_label = load_point_label

    def point_label_transform(self, point_label, resize, resize_dims, crop, flip, rotate):
        H, W = resize_dims
        point_label[:, :2] = point_label[:, :2] * resize
        point_label[:, 0] -= crop[0]
        point_label[:, 1] -= crop[1]
        if flip:
            point_label[:, 0] = resize_dims[1] - point_label[:, 0]

        point_label[:, 0] -= W / 2.0
        point_label[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        point_label[:, :2] = np.matmul(rot_matrix, point_label[:, :2].T).T

        point_label[:, 0] += W / 2.0
        point_label[:, 1] += H / 2.0

        coords = point_label[:, :2].astype(np.int16)

        depth_map = np.zeros(resize_dims)
        valid_mask = ((coords[:, 1] < resize_dims[0])
                    & (coords[:, 0] < resize_dims[1])
                    & (coords[:, 1] >= 0)
                    & (coords[:, 0] >= 0))
        depth_map[coords[valid_mask, 1],
                coords[valid_mask, 0]] = point_label[valid_mask, 2]
        semantic_map = np.zeros(resize_dims)
        semantic_map[coords[valid_mask, 1],
                coords[valid_mask, 0]] = (point_label[valid_mask, 3] >= 0)
        return torch.Tensor(depth_map), torch.Tensor(semantic_map)

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """
        Args:
            results:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
        """
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        gt_depth = []
        gt_semantic = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        # print(results['sat_path'])
        # if self.sat:
        #     sat_path = results['sat_path'] + '/sat.png'

        #     # for sat_path in results['sat_path']:/
        #     img_sat = Image.open(sat_path)
        #     # img_sat = self.img_transform_core(
        #     #     img_sat,
        #     #     resize_dims=resize_dims,
        #     #     crop=crop,
        #     #     flip=flip,
        #     #     rotate=rotate)
        #     imgs_sat.append(self.normalize_img(img_sat))

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            if self.load_point_label:
                point_filename = filename.replace('samples/', 'samples_point_label/'
                                                  ).replace('.jpg','.npy')
                point_label = np.load(point_filename).astype(np.float64)[:4].T
                point_depth_augmented, point_semantic_augmented = \
                    self.point_label_transform(
                            point_label, resize, self.data_config['input_size'],
                            crop, flip, rotate)
                gt_depth.append(point_depth_augmented)
                gt_semantic.append(point_semantic_augmented)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))    # 保存未归一化的图像，应该是为了做可视化.
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    # 对选择的邻近帧图像也进行增广, 增广参数与当前帧图像相同.
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))



            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        if self.sequential:
            for adj_info in results['adjacent']:
                # adjacent与current使用相同的图像增广, 相机内参也相同.
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                for cam_name in cam_names:
                    # 获得adjacent帧对应的camera2ego变换 (4, 4)和ego2global变换 (4, 4).
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)


        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        if self.load_point_label:
            gt_depth = torch.stack(gt_depth)
            gt_semantic = torch.stack(gt_semantic)
            results['gt_depth'] = gt_depth
            results['gt_semantic'] = gt_semantic
        results['canvas'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6

        # if self.sat:
            # return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, imgs_sat

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        # self.sat = if_sat

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        """
        Args:
            gt_boxes: (N, 9)
            rotate_angle:
            scale_ratio:
            flip_dx: bool
            flip_dy: bool

        Returns:
            gt_boxes: (N, 9)
            rot_mat: (3, 3）
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)     # 变换后的3D框中心坐标
            gt_boxes[:, 3:6] *= scale_ratio    # 变换后的3D框尺寸
            gt_boxes[:, 6] += rotate_angle     # 旋转后的3D框的方位角
            # 翻转也会进一步改变方位角
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']      # (N_gt, 9),  (N_gt, )
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        # gt_boxes: (N, 9)  BEV增广变换后的3D框
        # bda_rot: (3, 3)   BEV增广矩阵, 包括旋转、缩放和翻转.
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        # if self.sat:
        #     post_rots, post_trans, imgs_sat = results['img_inputs'][4:]
        # else:
        post_rots, post_trans = results['img_inputs'][4:6]
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                    post_trans, bda_rot)

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        # if 'voxel_semantics' in results:
        #     if flip_dx:
        #         results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
        #         results['mask_lidar'] = results['mask_lidar'][::-1, ...].copy()
        #         results['mask_camera'] = results['mask_camera'][::-1, ...].copy()
        #     if flip_dy:
        #         results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, ...].copy()
        #         results['mask_lidar'] = results['mask_lidar'][:, ::-1, ...].copy()
        #         results['mask_camera'] = results['mask_camera'][:, ::-1, ...].copy()

        return results


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

        # self.sat = if_sat

    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        # if self.sat:
            # post_rots, post_trans, bda, _ = results['img_inputs'][4:]
        # else:
        post_rots, post_trans, bda = results['img_inputs'][4:7]

        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            # lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)      # (N_points, 3):  3: (u, v, d)

            # 再考虑图像增广
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
            depth_map = self.points2depthmap(points_img,
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        semantics = torch.from_numpy(semantics)
        mask_lidar = torch.from_numpy(mask_lidar)
        mask_camera = torch.from_numpy(mask_camera)

        if results.get('flip_dx', False):
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

@PIPELINES.register_module()
class PrepareSATImageInputs(object):
    """卫星图像加载（表6/7/9 增强版，路径逻辑保持原始行为）。

    在原始版本基础上新增：
    - sat_mode: normal / random_same_scene / zero（表6 先验特异性）
    - jitter_tx/jitter_ty/jitter_yaw: 定位扰动（表7 鲁棒性）
    - sat_cache_dir: 离线特征缓存模式（表9）
    路径构建保持原始逻辑：/sat/ -> /{sat_dir}/ + '/sat.png'
    """
    def __init__(self, is_train=False, sequential=False,
                 sat_dirs=None,
                 test_sat_dir='sat_test',
                 color_jitter_prob=0.2,
                 color_jitter_strength=0.2,
                 sat_mode='normal',
                 random_seed=2025,
                 jitter_tx=0.0,
                 jitter_ty=0.0,
                 jitter_yaw=0.0,
                 jitter_random=False,
                 jitter_seed=2025,
                 sat_cache_dir=None):
        self.is_train = is_train
        self.sequential = sequential
        self.sat_dirs = sat_dirs if sat_dirs is not None else ['sat_test']
        self.test_sat_dir = test_sat_dir
        self.sat_cache_dir = sat_cache_dir  # 表9：离线特征缓存模式
        # 表7：训练时方向随机（幅度=配置值，方向均匀采样），测试时固定
        self.jitter_random = jitter_random
        self.color_jitter_prob = color_jitter_prob
        # 在原图归一化前做颜色抖动，保证全局一致性
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=color_jitter_strength * 0.1
        )
        # 表6：卫星输入模式
        assert sat_mode in ('normal', 'random_same_scene', 'zero'), \
            f'unknown sat_mode: {sat_mode}'
        self.sat_mode = sat_mode
        self.random_seed = random_seed
        # 表7：定位扰动参数（米/度）
        self.jitter_tx = jitter_tx
        self.jitter_ty = jitter_ty
        self.jitter_yaw = jitter_yaw
        self.jitter_seed = jitter_seed
        # 同场景随机替换缓存：scene_dir -> [token,...]；token -> 替换token
        self._scene_tokens_cache = {}
        self._replace_map = {}

    def _random_same_scene_path(self, sat_path):
        """同场景内随机替换另一帧的卫星图（表6）。
        固定种子 + token md5 作为 rng 种子，与样本处理顺序/进程数无关，跨运行可复现。
        """
        scene_dir = os.path.dirname(sat_path)
        token = os.path.basename(sat_path)
        if scene_dir not in self._scene_tokens_cache:
            tokens = sorted(
                t for t in os.listdir(scene_dir)
                if os.path.isdir(os.path.join(scene_dir, t)))
            self._scene_tokens_cache[scene_dir] = tokens
        tokens = self._scene_tokens_cache[scene_dir]
        if len(tokens) < 2:
            return sat_path
        if token not in self._replace_map:
            candidates = [t for t in tokens if t != token]
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            rng = random.Random((h ^ self.random_seed) & 0xFFFFFFFF)
            self._replace_map[token] = rng.choice(candidates)
        return os.path.join(scene_dir, self._replace_map[token])

    def _apply_jitter(self, img, sample_key=None):
        """定位扰动（表7）：平移（米→像素，0.2m/px）+ yaw 旋转（度）。
        归一化前对 PIL 图像施加。
        - jitter_random=False：tx/ty/yaw 取配置固定值
        - 训练且 jitter_random=True：每次在 ±配置值内均匀随机
        - 测试且 jitter_random=True：按 sample_key + jitter_seed 确定性均匀采样，
          与样本处理顺序/worker 数无关，并保证不同模型使用同一组扰动
        """
        if self.jitter_tx == 0.0 and self.jitter_ty == 0.0 and self.jitter_yaw == 0.0:
            return img
        if self.jitter_random:
            if self.is_train:
                rng = random
            else:
                assert sample_key is not None, '测试期随机扰动需要稳定的 sample_key'
                digest = hashlib.md5(
                    ('%s:%s' % (self.jitter_seed, sample_key)).encode()
                ).hexdigest()
                rng = random.Random(int(digest[:16], 16))
            tx = rng.uniform(-self.jitter_tx, self.jitter_tx)
            ty = rng.uniform(-self.jitter_ty, self.jitter_ty)
            yaw = rng.uniform(-self.jitter_yaw, self.jitter_yaw)
        else:
            tx, ty, yaw = self.jitter_tx, self.jitter_ty, self.jitter_yaw
        from torchvision.transforms.functional import affine
        px_per_m = 1.0 / 0.2
        tx_px = tx * px_per_m
        ty_px = ty * px_per_m
        return affine(
            img,
            angle=-yaw,              # 逆时针为正（图像坐标系）
            translate=(tx_px, ty_px),
            scale=1.0,
            shear=0,
            interpolation=Image.BILINEAR,
        )

    def _load_sat(self, sat_path_dir, jitter_fn):
        """加载卫星图 → 可选颜色抖动/定位扰动 → 归一化。
        支持 sat_mode: normal / random_same_scene / zero（表6）。
        """
        if self.sat_mode == 'zero':
            # 全零卫星输入：跳过文件读取（表6）
            return torch.zeros(3, 400, 400)
        if self.sat_mode == 'random_same_scene':
            sat_path_dir = self._random_same_scene_path(sat_path_dir)
        img_pil = Image.open(sat_path_dir + '/sat.png')
        if jitter_fn is not None:
            img_pil = jitter_fn(img_pil)   # 归一化前做颜色抖动
        img_pil = self._apply_jitter(img_pil, sample_key=sat_path_dir)
        return mmlabNormalize_sat(img_pil)

    def _pick_sat_dir(self):
        if self.is_train and len(self.sat_dirs) > 1:
            return np.random.choice(self.sat_dirs)
        return self.test_sat_dir  # 测试时固定用 sat_test

    def _build_sat_path(self, sat_path_raw, sat_dir):
        """保持原始路径逻辑：/sat/ -> /{sat_dir}/ + '/sat.png'"""
        return sat_path_raw.rstrip('/').replace('/sat/', f'/{sat_dir}/') + '/sat.png'

    def __call__(self, results):
        sat_dir = self._pick_sat_dir()
        # sequential 帧用同一个 jitter_fn：保证时序颜色变化一致，时序 loss 不受干扰
        jitter_fn = (self.color_jitter
                     if self.is_train and random.random() < self.color_jitter_prob
                     else None)

        # token 目录路径（不含 sat.png，保持与原始逻辑一致的构建方式）
        sat_path_dir = results['sat_path'].rstrip('/').replace('/sat/', f'/{sat_dir}/')
        if self.sat_cache_dir is not None:
            # 表9 离线特征缓存模式：跳过读图/归一化/扰动，直接加载预提取的 d0 特征
            token = os.path.basename(sat_path_dir)   # 末级即 sample token
            arr = np.load(os.path.join(self.sat_cache_dir, f'{token}.npz'))['d0']
            # Preserve float16 caches on CPU so host-to-device transfer is
            # actually halved. The detector casts to its compute dtype after
            # the tensor reaches the GPU. Float32 caches remain unchanged.
            img_sat = torch.from_numpy(arr)   # (128, 200, 200)
        else:
            img_sat = self._load_sat(sat_path_dir, jitter_fn)
            img_sat = torch.flip(img_sat, [1])
            if results.get('flip_dx', False):
                img_sat = torch.flip(img_sat, [2])
            if results.get('flip_dy', False):
                img_sat = torch.flip(img_sat, [1])

        imgs = [img_sat]

        if self.sequential:
            assert 'adjacent' in results
            for adj_info in results['adjacent']:
                adj_dir = adj_info['sat_path'].rstrip('/').replace('/sat/', f'/{sat_dir}/')
                img_adj = self._load_sat(adj_dir, jitter_fn)
                img_adj = torch.flip(img_adj, [1])
                if results.get('flip_dx', False):
                    img_adj = torch.flip(img_adj, [2])
                if results.get('flip_dy', False):
                    img_adj = torch.flip(img_adj, [1])
                imgs.append(img_adj)
            # 修复：stack后shape为 (N_frames, C, H, W)
            # DataLoader会自动加batch维度变成 (B, N_frames, C, H, W)
            imgs = torch.stack(imgs)  # (N_frames, C, H, W)
            results['img_inputs'] = tuple(list(results['img_inputs']) + [imgs])
        else:
            # 单帧模式：(C, H, W)，DataLoader加batch后变成 (B, C, H, W)
            results['img_inputs'] = tuple(list(results['img_inputs']) + imgs)

        return results
