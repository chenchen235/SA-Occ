# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from .ego_pose_dataset import EgoPoseDataset
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader
from ..core.evaluation.ray_metrics import main_raypq
import torch
import glob
import json
from ..core.evaluation.dynamic_mask_metrics import DynamicMaskMetrics


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
class NuScenesDatasetOccpancy(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def _save_nuscenes_like(self, info, vis_dump, show_dir):
        """
        保存为:
        show_dir/scene-xxx/CAM_XXX/<token>/{semantic|depth|recon}.jpg
        不用 meta，token 从 info['token'] 取；相机顺序按 nuScenes 常用顺序。
        """
        cam_order = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]
        cams_available = [c for c in cam_order if c in info['cams']]

        # scene 与 token
        scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem]
        scene_name = scene_name[0] if len(scene_name) > 0 else info.get('scene_name', 'unknown_scene')
        token = info['token']

        # 取张量
        sem = vis_dump['semantic_maps']   # [B,N,C,H,W] (cpu)
        dep = vis_dump['depth_maps']      # [B,N,H,W]   (cpu)
        rec = vis_dump['recon_imgs_srgb'] # [B,N,3,H,W] or [B,N,H,W,3] (cpu)

        # 颜色映射（7 类）
        semantic_colors = np.array([
            [255,   0,   0],   # 0 driveable
            [  0, 255,   0],   # 1 other_flat
            [  0,   0, 255],   # 2 sidewalk
            [255, 255,   0],   # 3 terrain
            [255,   0, 255],   # 4 manmade
            [  0, 255, 255],   # 5 vegetation
            [128, 128, 128],   # 6 air
        ], dtype=np.uint8)

        # 反标准化参数
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        std  = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

        B, N = dep.shape[:2]
        assert B == 1, "vis_dump 仅支持 batch=1 的推理保存"
        b = 0

        # 语义 argmax
        sem_idx = sem.argmax(axis=2)  # [B,N,H,W] -> np

        # 对齐相机数量
        use_N = min(N, len(cams_available))

        for n in range(use_N):
            cam_name = cams_available[n]
            cam_token_dir = os.path.join(show_dir, scene_name, cam_name, token)
            os.makedirs(cam_token_dir, exist_ok=True)

            # semantic
            sem_color = semantic_colors[sem_idx[b, n].astype(np.int64).clip(0, 6)]  # [H,W,3]
            Image.fromarray(sem_color).save(os.path.join(cam_token_dir, 'semantic.jpg'))
            print(f"Saved semantic map for {cam_name} at {cam_token_dir}")

            # depth
            dep_u8 = (dep[b, n].clip(0, 45) / 45.0 * 255.0).astype(np.uint8)
            Image.fromarray(dep_u8).save(os.path.join(cam_token_dir, 'depth.jpg'))

            # recon: [3,H,W] or [H,W,3]
            recon_bn = rec[b, n]
            if recon_bn.ndim == 3 and recon_bn.shape[0] == 3:
                rec_np = recon_bn.transpose(1, 2, 0)  # [H,W,3]
                rec_np = (rec_np * std + mean).clip(0, 255).astype(np.uint8)
            elif recon_bn.ndim == 3 and recon_bn.shape[-1] == 3:
                rec_np = (recon_bn * std + mean).clip(0, 255).astype(np.uint8)
            else:
                arr = recon_bn
                if arr.ndim == 2:
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                rec_np = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
            Image.fromarray(rec_np).save(os.path.join(cam_token_dir, 'recon.jpg'))

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric = eval_kwargs['metric'][0]
        print("metric = ", metric)
        if metric == 'ray-iou':
            occ_gts = []
            occ_preds = []
            lidar_origins = []
            inst_gts = []
            inst_preds = []

            print('\nStarting Evaluation...')

            data_loader = DataLoader(
                EgoPoseDataset(self.data_infos),
                batch_size=1,
                shuffle=False,
                num_workers=8
            )

            sample_tokens = [info['token'] for info in self.data_infos]

            for i, batch in enumerate(data_loader):
                # if i > 5:
                #     break
                token = batch[0][0]
                output_origin = batch[1]

                data_id = sample_tokens.index(token)
                info = self.data_infos[data_id]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                # occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'].replace('data/nuscenes/gts/', 'data/nuscenes/occ3d_panoptic/'), 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                occ_pred = occ_results[data_id]['pred_occ'].cpu().numpy()     # (Dx, Dy, Dz)
                # occ_pred = occ_results[data_id]['pred_occ']     # (Dx, Dy, Dz)

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(occ_pred)

                if 'pano_inst' in occ_results[data_id].keys():
                    pano_inst = occ_results[data_id]['pano_inst'].cpu()
                    # pano_inst = torch.from_numpy(occ_results[data_id]['pano_inst'])
                    pano_inst = pano_inst.squeeze(0).numpy()
                    gt_instances = occ_gt['instances']
                    inst_gts.append(gt_instances)
                    inst_preds.append(pano_inst)

            eval_results = calc_rayiou(occ_preds, occ_gts, lidar_origins)
            if len(inst_preds) > 0:
                eval_results.update(main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins))

        else:
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=True)
            dynamic_metrics = DynamicMaskMetrics()

            print('\nStarting Evaluation...')
            for index, occ_pred in enumerate(tqdm(occ_results)):
                info = self.data_infos[index]
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
                if isinstance(occ_pred, dict) and 'dynamic_prob' in occ_pred:
                    dynamic_metrics.update(occ_pred['dynamic_prob'], gt_semantics,
                                           mask_camera)

                self.occ_eval_metrics.add_batch(
                    occ_pred['pred_occ'] if (isinstance(occ_pred, dict) and 'pred_occ' in occ_pred) else occ_pred,
                    gt_semantics, mask_lidar, mask_camera
                )

                # 原有 npz 保存
                if show_dir is not None:
                    mmcv.mkdir_or_exist(show_dir)
                    scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                    sample_token = info['token']
                    mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                    save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                    np.savez_compressed(save_path,
                        pred=occ_pred['pred_occ'] if (isinstance(occ_pred, dict) and 'pred_occ' in occ_pred) else occ_pred,
                        gt=occ_gt, sample_token=sample_token)
                # print(1111111111111111111111111111111)
                # # 新增：按 nuScenes 相机目录保存 recon/semantic/depth
                # if isinstance(occ_pred, dict) and 'vis_dump' in occ_pred:
                #     print(22222222222222222222222222222222)
                #     self._save_nuscenes_like(info, occ_pred['vis_dump'], show_dir='./vis_test')

            eval_results = self.occ_eval_metrics.count_miou()
            dynamic_results = dynamic_metrics.summary()
            print('Dynamic-mask metrics:', dynamic_results)
            metric_path = os.environ.get('DYNAMIC_METRIC_JSON')
            if metric_path:
                with open(metric_path, 'w') as f:
                    json.dump(dynamic_results, f, indent=2, sort_keys=True)
            eval_results.update(dynamic_results)

        return eval_results


    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis
