from mmdet.models import DETECTORS
import torch
from .dualbev import DualBEV
import torch
from torchvision.transforms import Resize
import torchvision.transforms as transforms
import torch.nn.functional as F
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import torch.nn as nn
from ..necks.view_transformer_sat_dual import BEVGeomAttention   # ✅ 复用几何注意力
from .bevdet import inverse_rigid_transform

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, num_classes=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )
        # self.num_classes = num_classes

    def forward(self, score, target):
        # target_one_hot = F.one_hot(target, num_classes=self.num_classes).to(score.dtype)
        loss = self.criterion(score, target)
        return loss


def label_ce(masks_preds, true_masks, ignore=-1, num_classes=None):
    if ignore is None:
        criterion_ce = CrossEntropy(weight=None, num_classes=num_classes)
    else:
        criterion_ce = CrossEntropy(ignore_label=ignore, weight=None, num_classes=num_classes)
    ce_loss = criterion_ce(masks_preds, true_masks.long())
    return ce_loss

class OccFusionHead(nn.Module):
    """占据融合模块：BEV占据 + 卫星静态占据（2D卷积实现）"""
    def __init__(self, num_classes=18, static_classes=7, Dz=16):
        super().__init__()
        self.num_classes = num_classes  # 18
        self.static_classes = static_classes  # 7
        self.Dz = Dz

        # 用2D卷积处理：将Z维度合并到通道维度
        # 输入: (num_classes + static_classes) * Dz 通道
        # 输出: static_classes * Dz 通道的权重
        total_in_channels = (num_classes + static_classes) * Dz
        weight_out_channels = static_classes * Dz

        self.fusion_weight = nn.Sequential(
            nn.Conv2d(total_in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, weight_out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 细化网络
        refine_in_channels = num_classes * Dz
        self.refine_net = nn.Sequential(
            nn.Conv2d(refine_in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, refine_in_channels, kernel_size=3, padding=1),
        )

    def forward(self, occ_bev, occ_sat):
        """
        Args:
            occ_bev: (B, Dx, Dy, Dz, 18) BEV占据预测logits
            occ_sat: (B, Dx, Dy, Dz, 7) 卫星静态占据预测logits
        Returns:
            fused_occ: (B, Dx, Dy, Dz, 18) 融合后的占据预测
        """
        B, Dx, Dy, Dz, _ = occ_bev.shape

        # 转换为2D格式: (B, Dx, Dy, Dz, C) -> (B, C*Dz, Dx, Dy)
        occ_bev_2d = occ_bev.permute(0, 4, 3, 1, 2).reshape(B, self.num_classes * Dz, Dx, Dy)
        occ_sat_2d = occ_sat.permute(0, 4, 3, 1, 2).reshape(B, self.static_classes * Dz, Dx, Dy)

        # 计算融合权重
        concat_feat = torch.cat([occ_bev_2d, occ_sat_2d], dim=1)  # (B, 25*Dz, Dx, Dy)
        weight = self.fusion_weight(concat_feat)  # (B, 7*Dz, Dx, Dy)

        # 分离BEV的动态和静态部分
        occ_dynamic_2d = occ_bev_2d[:, :11 * Dz, :, :]  # (B, 11*Dz, Dx, Dy)
        occ_static_bev_2d = occ_bev_2d[:, 11 * Dz:, :, :]  # (B, 7*Dz, Dx, Dy)

        # 加权融合静态部分
        fused_static_2d = weight * occ_static_bev_2d + (1 - weight) * occ_sat_2d

        # 合并
        fused_occ_2d = torch.cat([occ_dynamic_2d, fused_static_2d], dim=1)  # (B, 18*Dz, Dx, Dy)

        # 细化
        fused_occ_2d = fused_occ_2d + self.refine_net(fused_occ_2d)

        # 转回3D格式: (B, C*Dz, Dx, Dy) -> (B, Dx, Dy, Dz, C)
        fused_occ = fused_occ_2d.reshape(B, self.num_classes, Dz, Dx, Dy)
        fused_occ = fused_occ.permute(0, 3, 4, 2, 1).contiguous()  # (B, Dx, Dy, Dz, 18)

        return fused_occ


@DETECTORS.register_module()
class SA_OCC(DualBEV):
    def __init__(self, *args, upsample=False, ifsat=True, sat_img_backbone=None,
                 occ_head=None, sat_static_occ_head=None, use_occ_fusion=True,
                 use_sat_consist=False, sat_consist_weight=0.1,
                 sat_consist_cam_ch=80, sat_consist_sat_ch=80,
                 use_neck_sat_fuse=False,          # ✅ neck多尺度融合开关
                 neck_stage_sat_chs=(128, 128),   # ✅ d1/d2 通道
                 neck_stage_cam_chs=(160, 320),   # ✅ backbone前两stage
                 sat_down_ratio=1.0,
                 use_sat_cache=False,             # ✅ 表9：离线特征缓存模式（跳过 sat_img_backbone）
                 dynamic_mask_target='union',
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.sat = ifsat
        self.sat_down_ratio = sat_down_ratio
        self.use_sat_cache = use_sat_cache
        valid_dynamic_targets = {'occupancy', 'boxes', 'union'}
        if dynamic_mask_target not in valid_dynamic_targets:
            raise ValueError(
                f'Unsupported dynamic_mask_target={dynamic_mask_target!r}; '
                f'expected one of {sorted(valid_dynamic_targets)}')
        self.dynamic_mask_target = dynamic_mask_target
        if ifsat:
            self.sat_img_backbone = builder.build_backbone(sat_img_backbone)
        else:
            self.sat_img_backbone = None

        self.occ_head = builder.build_head(occ_head)

        if sat_static_occ_head is not None:
            self.sat_static_occ_head = builder.build_head(sat_static_occ_head)
        else:
            self.sat_static_occ_head = None

        # ✅ 新增：占据融合模块
        self.use_occ_fusion = use_occ_fusion
        if self.sat_static_occ_head is not None and use_occ_fusion:
            self.occ_fusion = OccFusionHead(num_classes=18, static_classes=7)
        else:
            self.occ_fusion = None

        self.pts_bbox_head = None
        self.upsample = upsample
        self.count = 0
        self.use_sat_consist = use_sat_consist   # ✅
        self.sat_consist_weight = sat_consist_weight
        if use_sat_consist and sat_consist_cam_ch is not None and sat_consist_sat_ch is not None:
            C = min(sat_consist_cam_ch, sat_consist_sat_ch)
            self.sat_consist_proj_cam = nn.Sequential(
                nn.Conv2d(sat_consist_cam_ch, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 1, bias=False),
            )
            self.sat_consist_proj_sat = nn.Conv2d(sat_consist_sat_ch, C, 1, bias=False)
            nn.init.orthogonal_(self.sat_consist_proj_cam[0].weight)
            nn.init.orthogonal_(self.sat_consist_proj_sat.weight)
        else:
            self.sat_consist_proj_cam = None
            self.sat_consist_proj_sat = None

        self.use_neck_sat_fuse = use_neck_sat_fuse
        if use_neck_sat_fuse:
            self.neck_sat_fusion = NeckSATMultiScaleFusion(
                stage_sat_chs=neck_stage_sat_chs, stage_cam_chs=neck_stage_cam_chs)
        else:
            self.neck_sat_fusion = None

        def _check_grad(module, name_prefix=''):
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                if param.requires_grad and param.grad is None:
                    print(f"[no grad] {full_name}")
            for child_name, child in module.named_children():
                _check_grad(child, f"{name_prefix}.{child_name}" if name_prefix else child_name)

        # # 只在第一个 backward 后触发一次
        # self._grad_check_done = False
        # def _grad_hook(grad):
        #     if not self._grad_check_done:
        #         self._grad_check_done = True
        #         _check_grad(self)
        # # 挂在任意一个肯定有梯度的参数上
        # list(self.parameters())[0].register_hook(_grad_hook)

    @force_fp32()
    def satbev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_satbev_encoder_backbone(x)
        x = self.img_satbev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def image_encoder(self, img, img_sat, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        sat_multiscale = None
        if self.sat:
            if self.use_sat_cache:
                # ✅ 表9 离线特征缓存模式：img_sat 已是 d0 特征 (B,128,200,200)，跳过 sat_img_backbone
                # A float16 cache is transferred at half the bandwidth cost,
                # then restored to the model compute dtype on GPU.
                x_sat = img_sat.to(dtype=imgs.dtype)
                if self.use_neck_sat_fuse:
                    # d1 由 d0 2×2 avg-pool 近似（在线 d1 为解码器 1/4 尺度输出，等价下采样）
                    sat_multiscale = [x_sat, F.avg_pool2d(x_sat, 2)]
            else:
                # ✅ 低分辨率测试: 直接降采样输入到 sat_down_ratio, 降低 backbone 计算量
                orig_h, orig_w = img_sat.shape[-2:]
                if self.sat_down_ratio < 1.0:
                    down_h = int(round(orig_h * self.sat_down_ratio))
                    down_w = int(round(orig_w * self.sat_down_ratio))
                    img_sat_in = F.interpolate(img_sat, size=(down_h, down_w),
                                               mode='bilinear', align_corners=False)
                else:
                    img_sat_in = img_sat

                if self.use_neck_sat_fuse:
                    # ✅ 需要 UNet 多尺度
                    x_sat, sat_multiscale = self.sat_img_backbone(img_sat_in, return_multiscale=True)
                else:
                    x_sat = self.sat_img_backbone(img_sat_in)

                # ✅ backbone 输出随输入等比缩小, 上采样回原始输入对应的输出尺寸
                #    ResNetUNet18: d0@1/2, d1@1/4, 保持下游 SAT 融合与 BEV(200×200) 对齐
                if self.sat_down_ratio < 1.0:
                    target_d0 = (orig_h // 2, orig_w // 2)
                    x_sat = F.interpolate(x_sat, size=target_d0,
                                          mode='bilinear', align_corners=True)
                    if sat_multiscale is not None:
                        target_d1 = (orig_h // 4, orig_w // 4)
                        sat_multiscale = [
                            F.interpolate(sat_multiscale[0], size=target_d0,
                                          mode='bilinear', align_corners=True),
                            F.interpolate(sat_multiscale[1], size=target_d1,
                                          mode='bilinear', align_corners=True),
                        ]
        else:
            x_sat = None
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat, x_sat, img_sat, sat_multiscale   # ✅ 多返回一个

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        # print("prepare: ", B, N, C, H, W)
        if self.sat:
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda, img_sat = inputs
        else:
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = inverse_rigid_transform(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        if self.sat:
            return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda, img_sat]
        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, **kwargs):
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img[1:7]
        mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)
        sat_multiscale = None
        if self.sat:
            x, _, x_sat, img_sat, sat_multiscale = self.image_encoder(img[0], img[7])
            inputs=[x, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input, x_sat]
        else:
            x, _, _, _, _ = self.image_encoder(img[0], "_")
            inputs=[x, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input]

        vt_out = self.img_view_transformer(inputs)
        if len(vt_out) == 9:
            x, depth, bev_mask, semantic, sat_sem, sat_height, projected_sem_logit, sat_static_feat, bev_feat_cam = vt_out
        else:
            x, depth, bev_mask, semantic, sat_sem, sat_height, projected_sem_logit, sat_static_feat = vt_out
            bev_feat_cam = None

        # ✅ neck 多尺度融合：拆 backbone+neck，逐尺度注入SAT，FPN后一次geom_att
        # 200注入d0 → backbone → 100/50/25注入d1/d2/d3
        if self.use_neck_sat_fuse and self.neck_sat_fusion is not None and sat_multiscale is not None:
            bb_feats = self.img_bev_encoder_backbone(x)
            # ✅ sat_multiscale = [d0@200, d1@100, d2@50, d3@25]
            # inject_indices=(0,2) → feat[0]注入d1, feat[2]注入d3，跳过d2(FPN不用feat[1])
            sat_for_inject = [sat_multiscale[1]]   # ✅ 只用 d1@100，去掉 d2/d3
            bb_feats = self.neck_sat_fusion.inject_stages(
                bb_feats, sat_for_inject, projected_sem_logit)
            x = self.img_bev_encoder_neck(bb_feats)
            if isinstance(x, (list, tuple)):
                x = x[0]
        else:
            x = self.bev_encoder(x)
        return [x], depth, bev_mask, semantic, sat_sem, sat_height, img_sat, projected_sem_logit, sat_static_feat, bev_feat_cam

    def extract_feat(self, points, img, img_metas, **kwargs):
        img_feats, depth, bev_mask, semantic, sem_sat, height_sat, img_sat, projected_sem_logit, sat_static_feat, bev_feat_cam = \
            self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, bev_mask, semantic, sem_sat, height_sat, img_sat, projected_sem_logit, sat_static_feat, bev_feat_cam)

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                      gt_labels_3d=None, gt_labels=None, gt_bboxes=None,
                      img_inputs=None, proposals=None, gt_bboxes_ignore=None, **kwargs):
        img_feats, pts_feats, depth, bev_mask, semantic, sem_sat, height_sat, img_sat, \
            projected_sem_logit, sat_static_feat, bev_feat_cam = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)

        # print('projected_sem_logit1111111: ', projected_sem_logit.shape)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        loss_depth, loss_ce_semantic = \
            self.img_view_transformer.get_loss(depth, semantic, gt_depth, gt_semantic)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        # 生成静态占据GT
        static_voxel_semantics = torch.where((voxel_semantics >= 11) & (voxel_semantics <= 16),
                                               voxel_semantics.clone() - 11, torch.full_like(voxel_semantics, 6))

        # # ✅ 动态类(0-10) → -1(ignore)，静态类(11-16) → 0-5，free(17) → 6
        # static_voxel_semantics = torch.zeros_like(voxel_semantics)   # 默认0，safe dummy
        # static_mask = (voxel_semantics >= 11) & (voxel_semantics <= 16)
        # free_mask = (voxel_semantics == 17)
        # dynamic_mask = (voxel_semantics <= 10)                        # ✅ 动态类 mask
        # static_voxel_semantics[static_mask] = voxel_semantics[static_mask] - 11
        # static_voxel_semantics[free_mask] = 6
        # static_voxel_semantics[dynamic_mask] = 6
        # # dynamic 保持 0，但后面 sat_occ_mask 会把它们全部排除

        # # B_occ, Dx, Dy, Dz = voxel_semantics.shape
        # # dev = voxel_semantics.device
        # valid_cat = (voxel_semantics >= 11) & (voxel_semantics <= 16)          # [B,Dx,Dy,Dz]
        # # has_valid  = valid_cat.any(dim=3)                                       # [B,Dx,Dy]
        # # has_cam    = mask_camera.bool().any(dim=3)                              # [B,Dx,Dy]
        # # z_exp = torch.arange(Dz, device=dev).view(1, 1, 1, Dz).expand(B_occ, Dx, Dy, Dz)

        # # z_surface = torch.where(valid_cat, z_exp, torch.full_like(z_exp, -1)).max(3, keepdim=True)[0]
        # # z_cam_max = torch.where(mask_camera.bool(), z_exp, torch.full_like(z_exp, -1)).max(3, keepdim=True)[0]

        # # voxel_at_cam = voxel_semantics.gather(3, z_cam_max.clamp(0, Dz - 1).long())
        # # sem_at_surf  = voxel_semantics.gather(3, z_surface.clamp(0, Dz - 1).long())

        # # surf_is_ground = has_valid.unsqueeze(-1) & (sem_at_surf >= 11) & (sem_at_surf <= 14)
        # # cam_bd_is_air  = (~((voxel_at_cam >= 11) & (voxel_at_cam <= 16))) & surf_is_ground
        # # above_air = (z_exp > z_cam_max) & has_cam.unsqueeze(-1) & cam_bd_is_air

        # # building_col = (voxel_semantics == 15).any(dim=3)                       # [B,Dx,Dy]
        # # z_man_max = torch.where((voxel_semantics == 15), z_exp, torch.full_like(z_exp, -1)).max(3, keepdim=True)[0]
        # # false_air_building = (z_exp > z_man_max) & building_col.unsqueeze(-1) & ~valid_cat

        # # no_cam_ground_air = (z_exp > z_surface + 1) & ~has_cam.unsqueeze(-1) & surf_is_ground

        # # sat_occ_mask = (
        # #     mask_camera.bool() | valid_cat | above_air | no_cam_ground_air
        # # ) & ~false_air_building \
        # #   & ~dynamic_mask           # ✅ 显式排除动态类，不再用 >=0 判断
        # sat_occ_mask = (
        #     mask_camera.bool() | valid_cat #| above_air | no_cam_ground_air
        # ) #& ~dynamic_mask           # ✅ 显式排除动态类，不再用 >=0 判断
        # sat_occ_mask = sat_occ_mask.to(mask_camera.dtype)

        occupancy_dynamic_mask = (
            (voxel_semantics >= 0) & (voxel_semantics <= 10)
        ).any(dim=-1).permute(0, 2, 1)
        boxes_dynamic_mask = kwargs['gt_bev_mask'].bool()
        if self.dynamic_mask_target == 'occupancy':
            gt_bev_mask = occupancy_dynamic_mask
        elif self.dynamic_mask_target == 'boxes':
            gt_bev_mask = boxes_dynamic_mask
        else:
            gt_bev_mask = occupancy_dynamic_mask | boxes_dynamic_mask

        position_encoding = torch.arange(voxel_semantics.size(3), dtype=torch.long).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(voxel_semantics.device)
        position_encoding = position_encoding.expand_as(voxel_semantics)
        categories = torch.arange(11, 17, dtype=torch.long).to(voxel_semantics.device)

        valid_category_mask = torch.stack([voxel_semantics == category for category in categories], dim=0).any(dim=0).to(voxel_semantics.device)

        valid_mask = torch.any(valid_category_mask, dim=3).permute(0, 2, 1)
        position_encoded_mask = position_encoding * valid_category_mask
        dz_max_indices = torch.argmax(position_encoded_mask, dim=-1).to(voxel_semantics.device).permute(0, 2, 1)

        sat_semantic_gt = voxel_semantics.permute(0, 2, 1, 3).gather(3, dz_max_indices.unsqueeze(-1).long()).squeeze(-1)
        sat_semantic_gt = torch.where(valid_mask, sat_semantic_gt-10, torch.zeros_like(sat_semantic_gt))

        sat_semantic_gt = sat_semantic_gt.view(-1)[valid_mask.reshape(-1)]
        sat_dz_gt = dz_max_indices.reshape(-1)[valid_mask.reshape(-1)]

        loss_sat_height = self.img_view_transformer.get_sat_depth_loss(sat_dz_gt, height_sat, valid_mask.reshape(-1))
        loss_sat_sem = self.img_view_transformer.get_sat_sem_loss(sat_semantic_gt, sem_sat, valid_mask.reshape(-1))

        losses = dict(loss_depth=loss_depth, loss_ce_semantic=loss_ce_semantic,
                      loss_sat_height=loss_sat_height, loss_sat_sem=loss_sat_sem)

        loss_projected_sem = self.img_view_transformer.get_projected_sem_loss(
            gt_bev_mask, projected_sem_logit)
        losses.update(loss_projected_sem)

        occ_bev_feature = img_feats[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2, mode='bilinear', align_corners=True)
            sat_static_feat = F.interpolate(sat_static_feat, scale_factor=2, mode='bilinear', align_corners=True)

        if self.use_sat_consist and bev_feat_cam is not None:
            static_bev_mask = ((voxel_semantics >= 11) & (voxel_semantics <= 16)).any(dim=-1)
            static_bev_mask_hw = static_bev_mask.permute(0, 2, 1).float()
            cam_feat = bev_feat_cam.detach()
            sat_feat = sat_static_feat
            if sat_feat.shape[-2:] != cam_feat.shape[-2:]:
                sat_feat = F.interpolate(sat_feat, size=cam_feat.shape[-2:],
                                         mode='bilinear', align_corners=False)
            mask_hw = F.interpolate(static_bev_mask_hw.unsqueeze(1),
                                    size=cam_feat.shape[-2:],
                                    mode='nearest').squeeze(1)
            cam_proj = self.sat_consist_proj_cam(cam_feat)
            sat_proj = self.sat_consist_proj_sat(sat_feat)
            cam_n = F.normalize(cam_proj, dim=1)
            sat_n = F.normalize(sat_proj, dim=1)
            cos_dist = 1.0 - (cam_n * sat_n).sum(dim=1)
            loss_sat_consist = (cos_dist * mask_hw.float()).sum() / \
                               mask_hw.float().sum().clamp(min=1.0)
            losses['loss_sat_consist'] = self.sat_consist_weight * loss_sat_consist       # 权重 0.1，不压主 loss

        # ✅ 核心修改：带融合的占据训练
        if self.sat_static_occ_head is not None:
            # 获取两个分支的输出
            outs_bev = self.occ_head(occ_bev_feature)      # (B, 18, Dx, Dy, Dz)
            outs_sat = self.sat_static_occ_head(sat_static_feat)  # (B, 7, Dx, Dy, Dz)

            if self.occ_fusion is not None:
                # 融合两个分支
                outs_fused = self.occ_fusion(outs_bev, outs_sat)

                # 主loss：融合结果监督全部18类
                loss_occ = self.occ_head.loss(outs_fused, voxel_semantics, mask_camera)
                losses.update(loss_occ)

                # BEV分支辅助loss（可选，帮助BEV分支独立学习）
                loss_occ_bev_aux = self.occ_head.loss(outs_bev, voxel_semantics, mask_camera)
                loss_occ_bev_aux = {f'bev_aux_{k}': v * 0.25 for k, v in loss_occ_bev_aux.items()}
                losses.update(loss_occ_bev_aux)
            else:
                # 不使用融合，只用BEV
                loss_occ = self.occ_head.loss(outs_bev, voxel_semantics, mask_camera)
                losses.update(loss_occ)

            # 卫星静态分支辅助loss
            # occ_mask = (mask_camera.bool() | valid_category_mask.bool()).to(mask_camera.dtype)
            loss_sat_static_occ = self.sat_static_occ_head.loss(
                outs_sat, static_voxel_semantics, mask_camera)
            loss_sat_static_occ = {f'sat_static_{k}': v * 0.25 for k, v in loss_sat_static_occ.items()}
            losses.update(loss_sat_static_occ)
        else:
            # 没有卫星分支
            loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
            losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def forward_sat_static_occ_train(self, sat_static_feat, static_voxel_semantics, mask_camera):
        """
        卫星静态占据训练（只监督静态类别0-5 + 空气6）
        Args:
            sat_static_feat: (B, C, Dy, Dx)
            static_voxel_semantics: (B, Dx, Dy, Dz)
                - 0-5: 静态类别（原11-16）
                - 6: 空气（原17）
                - -1: ignore（原0-10动态类别）
            mask_camera: (B, Dx, Dy, Dz)
        """
        outs = self.sat_static_occ_head(sat_static_feat)  # (B, 7, Dx, Dy, Dz)

        loss_sat_static_occ = self.sat_static_occ_head.loss(
            outs,
            static_voxel_semantics,  # 动态类别=-1会被ignore
            mask_camera,
        )# * 0.5

        # 添加前缀区分损失
        loss_sat_static_occ = {f'sat_static_{k}': v * 0.25 for k, v in loss_sat_static_occ.items()}
        return loss_sat_static_occ

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        img_feats, _, _, _, _, sem_sat, height_sat, img_sat, projected_sem_logit, sat_static_feat, _ = \
            self.extract_feat(points, img=img, img_metas=img_metas, **kwargs)  # ✅ 11个值

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
            sat_static_feat = F.interpolate(sat_static_feat, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        # ✅ 测试时使用融合
        if self.sat_static_occ_head is not None and self.occ_fusion is not None:
            outs_bev = self.occ_head(occ_bev_feature)
            outs_sat = self.sat_static_occ_head(sat_static_feat)
            outs_fused = self.occ_fusion(outs_bev, outs_sat)

            # 使用融合结果获取预测
            if not hasattr(self.occ_head, "get_occ_gpu"):
                occ_preds = self.occ_head.get_occ(outs_fused, img_metas)
            else:
                occ_preds = self.occ_head.get_occ_gpu(outs_fused, img_metas)
        else:
            occ_preds = self.simple_test_occ(occ_bev_feature, img_metas)

        dynamic_prob = torch.sigmoid(projected_sem_logit)
        return [dict(pred_occ=pred, dynamic_prob=dynamic_prob[b].detach().cpu().half())
                for b, pred in enumerate(occ_preds)]

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]

        return occ_preds

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _, _, _, _, _, _, _, _, sat_static_feat, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)  # ✅ 11个值
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)

        return outs

class NeckSATMultiScaleFusion(nn.Module):
    """动静解耦多尺度SAT融合：
    - 200尺度：注入 d0（对应原VT融合，保留高分辨率）
    - 100/50尺度：注入 d1/d2（backbone前两stage）
    - 25尺度：不注入（线性下采样太粗）
    - 无 post_geom_att（去掉）
    - 零初始化残差：起步=纯cam baseline
    """
    def __init__(self, stage_sat_chs=(128,), stage_cam_chs=(160,),
                 inject_indices=(0,)):    # ✅ 只注入 feat[0]@100
        super().__init__()
        self.inject_indices = inject_indices
        self.stage_inject = nn.ModuleList([
            self._zero_inject(s, c) for s, c in zip(stage_sat_chs, stage_cam_chs)
        ])

    @staticmethod
    def _zero_inject(in_ch, out_ch):
        conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        nn.init.constant_(conv[1].weight, 0.01)   # ✅ 改为小值，梯度不为零
        nn.init.zeros_(conv[1].bias)
        return conv

    def _inject(self, module, feat, sat, gate):
        if sat.shape[-2:] != feat.shape[-2:]:
            sat = F.interpolate(sat, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        # g = F.interpolate(gate, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        return feat + module(sat * gate)

    def inject_stages(self, feats, sat_feats, proj_logit):
        prob = torch.sigmoid(proj_logit)                          # (B,1,200,200)
        gate = 1 - prob                                           # 静态区≈1
        out = list(feats)
        for j, i in enumerate(self.inject_indices):
            if j < len(self.stage_inject) and i < len(feats) and j < len(sat_feats):
                size = feats[i].shape[-2:]
                # ✅ 用 max_pool 下采样 gate：动态区的 prob 最大值保留
                #    即动态区在低分辨率下仍接近1，gate仍接近0
                prob_ds = F.max_pool2d(prob,
                    kernel_size=(200 // size[0], 200 // size[1]),
                    stride=(200 // size[0], 200 // size[1]))
                gate_ds = 1 - prob_ds                             # 动态区严格为0
                out[i] = self._inject(self.stage_inject[j], feats[i], sat_feats[j], gate_ds)
        return out
