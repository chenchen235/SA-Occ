"""
Copyright (c) Zhijia Technology. All rights reserved.

Author: Peidong Li (lipeidong@smartxtruck.com / peidongl@outlook.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.utils import LearnedPositionalEncoding
from .view_transformer import DepthNet, LSSViewTransformerBEVDepth
# from ..model_utils import SAT_DepthNet
from ...ops.bev_pool_v2.bev_pool import bev_pool_v2
# from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import NECKS
# from .masked_seg_t import Fusion_Atten_Masked_Seg
import torchvision.transforms as transforms

class SATDeformAlign(nn.Module):
    """GPS漂移鲁棒的 SAT→cam BEV 可变形对齐。
    放在 conv_last/conv_last_sem/conv_last_height 之前，
    让现有的 sat_height/sat_sem/sat_static_occ loss 直接监督漂移学习。

    零初始化 offset：起步 aligned≈sat_feat，退化安全。
    dirac 初始化采样权重：起步近似恒等变换。
    """
    def __init__(self, cam_ch, sat_ch, deform_groups=4, max_offset=8.0,
                 use_mask=True):
        super().__init__()
        self.dg = deform_groups
        self.max_offset = max_offset
        self.use_mask = use_mask
        ks = 3
        self._ks = ks
        self._pad = ks // 2
        n_pts = deform_groups * ks * ks

        # cam(detach) + sat 拼接预测 offset；旧路径可继续预测 modulation mask。
        # 新的 SE(2)+deform 路径关闭 mask，使零 offset + dirac weight 严格恒等。
        out_channels = (3 if use_mask else 2) * n_pts
        self.offset_head = nn.Sequential(
            nn.Conv2d(cam_ch + sat_ch, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=True),
        )
        # ✅ 零初始化：起步 offset=0，退化为无对齐
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

        # 可变形采样权重
        self.align_weight = nn.Parameter(
            torch.zeros(sat_ch, sat_ch // deform_groups, ks, ks))
        # ✅ dirac 初始化：起步中心点权重=1，近似恒等
        nn.init.dirac_(self.align_weight, groups=deform_groups)

    def forward(self, cam_feat, sat_feat):
        """
        Args:
            cam_feat: (B, cam_ch, H, W) 纯相机 BEV，detach 作为参考系
            sat_feat: (B, sat_ch, H, W) SAT 特征，存在 GPS 漂移
        Returns:
            aligned: (B, sat_ch, H, W) 对齐后的 SAT 特征
        """
        n_pts = self.dg * self._ks * self._ks

        # cam detach：只提供"去哪里"的位置参考，不被 SAT 对齐污染
        x = torch.cat([cam_feat.detach(), sat_feat], dim=1)
        om = self.offset_head(x)

        # tanh 限幅：防止训练初期 offset 爆炸
        offset = self.max_offset * torch.tanh(om[:, :2 * n_pts])   # (B, 2*n_pts, H, W)
        mask = (torch.sigmoid(om[:, 2 * n_pts:])
                if self.use_mask else None)                         # (B, n_pts, H, W)

        aligned = deform_conv2d(
            sat_feat.float(),
            offset.float(),
            self.align_weight.float(),
            padding=self._pad,
            mask=(mask.float() if mask is not None else None),
        )
        if mask is None:
            # CUDA deform_conv 的零 offset 数值上并非逐位等于普通恒等映射。
            # 用同一算子的零 offset 结果作基线，既保证初始严格恒等，
            # 又保留 aligned 对预测 offset 的梯度。
            identity_aligned = deform_conv2d(
                sat_feat.float(),
                torch.zeros_like(offset, dtype=torch.float32),
                self.align_weight.float(),
                padding=self._pad,
                mask=None,
            )
            aligned = sat_feat.float() + (aligned - identity_aligned)
        return aligned.to(sat_feat.dtype)


class SATGlobalSE2Align(nn.Module):
    """从 camera BEV 与 SAT feature 预测单个全局 SE(2) 并粗对齐 SAT。

    输出层零初始化，因此初始 theta 是严格单位变换。平移以米为配置单位，
    内部根据 BEV cell size 转为 affine_grid 所需的归一化坐标。
    """
    def __init__(self, cam_ch, sat_ch, max_translation_m=8.0,
                 max_yaw_deg=15.0, meters_per_cell=0.4):
        super().__init__()
        self.max_translation_cells = max_translation_m / meters_per_cell
        self.max_yaw_rad = np.deg2rad(max_yaw_deg)

        in_ch = cam_ch + sat_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)
        self.last_se2 = None

    def forward(self, cam_feat, sat_feat):
        x = torch.cat([cam_feat.detach(), sat_feat], dim=1)
        raw = self.regressor(self.encoder(x)).float()
        tx_cells = self.max_translation_cells * torch.tanh(raw[:, 0])
        ty_cells = self.max_translation_cells * torch.tanh(raw[:, 1])
        yaw = self.max_yaw_rad * torch.tanh(raw[:, 2])

        _, _, h, w = sat_feat.shape
        tx_norm = 2.0 * tx_cells / max(w - 1, 1)
        ty_norm = 2.0 * ty_cells / max(h - 1, 1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        theta = torch.stack([
            cos_yaw, -sin_yaw, tx_norm,
            sin_yaw, cos_yaw, ty_norm,
        ], dim=1).view(-1, 2, 3)

        grid = F.affine_grid(theta, sat_feat.shape, align_corners=True)
        aligned = F.grid_sample(
            sat_feat.float(), grid, mode='bilinear', padding_mode='zeros',
            align_corners=True)
        identity_theta = theta.new_zeros(theta.shape)
        identity_theta[:, 0, 0] = 1.0
        identity_theta[:, 1, 1] = 1.0
        identity_grid = F.affine_grid(
            identity_theta, sat_feat.shape, align_corners=True)
        identity_aligned = F.grid_sample(
            sat_feat.float(), identity_grid, mode='bilinear',
            padding_mode='zeros', align_corners=True)
        aligned = sat_feat.float() + (aligned - identity_aligned)
        self.last_se2 = torch.stack([tx_cells, ty_cells, yaw], dim=1).detach()
        return aligned.to(sat_feat.dtype)


class SATSE2DeformAlign(nn.Module):
    """全局 SE(2) 粗配准 + 无 mask 的局部 deform refinement。"""
    def __init__(self, cam_ch, sat_ch, deform_groups=4,
                 local_max_offset=8.0, max_translation_m=8.0,
                 max_yaw_deg=15.0, meters_per_cell=0.4):
        super().__init__()
        self.global_align = SATGlobalSE2Align(
            cam_ch=cam_ch,
            sat_ch=sat_ch,
            max_translation_m=max_translation_m,
            max_yaw_deg=max_yaw_deg,
            meters_per_cell=meters_per_cell,
        )
        self.local_align = SATDeformAlign(
            cam_ch=cam_ch,
            sat_ch=sat_ch,
            deform_groups=deform_groups,
            max_offset=local_max_offset,
            use_mask=False,
        )

    def forward(self, cam_feat, sat_feat):
        sat_feat = self.global_align(cam_feat, sat_feat)
        return self.local_align(cam_feat, sat_feat)

class MS_CAM(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"
    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return self.sigmoid(xlg)


# B C H W -> B C 1 -> fc -C >
class ChannelAttention(nn.Module):

    def __init__(self, input_channel, output_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# B C H W -> B 1 H W -> conv 7x7 > sigmoid >

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes,planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ProbNet(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(ProbNet, self).__init__()
        self.loss_weight=loss_weight
        mid_channels=in_channels//2
        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.prob_conv = nn.Sequential(
            ResCBAMBlock(mid_channels, mid_channels),
            )
        self.mask_net = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        if with_centerness:
            self.centerness = bev_centerness_weight(bev_size[0],bev_size[1]).cuda()
        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        height_feat = self.base_conv(input)
        height_feat = self.prob_conv(height_feat)
        bev_prob = self.mask_net(height_feat)
        # print("bev_prob: ", bev_prob.shape)
        return bev_prob

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h,1)).mean()
        else:
            mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return dict(mask_ce_loss=self.loss_weight*mask_ce_loss, mask_dice_loss=mask_dice_loss)

class DualFeatFusion(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(DualFeatFusion, self).__init__()
        self.ca = MS_CAM(output_channel * 2, output_channel)
        # self.output_channel = output_channel
        # self.d1 = nn.Sequential(
        #     nn.Conv2d(input_channel//2, output_channel, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(output_channel),
        #     nn.ReLU(inplace=True),
        # )
        # self.d2 = nn.Sequential(
        #         nn.Conv2d(input_channel//2, output_channel, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(output_channel),
        #         nn.ReLU(inplace=True),
        #     )
    def forward(self, x1, x2):
        # x1 = self.d1(x1)
        # x2 = self.d2(x2)
        # x1 = x1[:, :self.output_channel, :, :]
        # x2 = x2[:, :self.output_channel, :, :]

        channel_factor = self.ca(torch.cat((x1,x2),1))
        out = channel_factor*x1 + (1-channel_factor)*x2
        return out

class BEVGeomAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(BEVGeomAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False) # 3—3

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bev_prob):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        # return self.sigmoid(x1)
        # bev_prob = self.conv2(bev_prob)
        return self.sigmoid(x1+bev_prob)


def bev_centerness_weight(nx, ny):
    xs, ys = torch.meshgrid(torch.arange(0, nx), torch.arange(0, nx))
    grid = torch.cat([xs[:, :, None], ys[:, :, None]], -1)
    grid = grid - nx//2
    grid = grid / (nx//2)
    centerness = (grid[..., 0]**2 + grid[..., 1]**2) / 2
    centerness = centerness.sqrt() + 1
    return centerness

class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, loss_weight=1.):
        super(DiceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.loss_weight=loss_weight

    def forward(self, inputs, targets, smooth=1e-5):
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return self.loss_weight*(1 - dice)

class Align(nn.Module):
    def __init__(self, inplane, kernel_size=3):
        super(Align, self).__init__()

        self.down_1 = nn.Conv2d(inplane, inplane, 1, bias=False)
        self.down_2 = nn.Conv2d(inplane, inplane, 1, bias=False)
        self.flow_make = nn.Conv2d(inplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)
        # self.reg = nn.Sequential(
        #     nn.Conv2d(inplane*2, 64, 3, 1, 1), nn.GroupNorm(8,64), nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, 1, 1),   nn.GroupNorm(8,64), nn.ReLU(),
        #     nn.Conv2d(64, 2,  3, 1, 1)   # Δx, Δy
        # )

    def forward(self, f_sat, f_bev):
        h, w = f_sat.size()[2:]
        size = (h, w)

        x1 = self.down_1(f_sat)
        x2 = self.down_2(f_bev)
        flow = self.flow_make(torch.cat([x1, x2], 1))
        # flow = self.reg(torch.cat([f_sat, f_bev], 1))  # B 2 H W
        f_sat_warp = self.flow_warp(f_sat, flow, size=size)
        return f_sat_warp


    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

@NECKS.register_module()
class DualViewTransformerFull_SAT(LSSViewTransformerBEVDepth):
    def __init__(self, pc_range, bev_h=200, bev_w=200, num_height=13, collapse_z=True, loss_semantic_weight=1,
                 depth_threshold=1, semantic_threshold=0.25, if_sat=True, depthnet_cfg=dict(),
                 use_sat_deform=False,
                 sat_deform_groups=4,
                 sat_deform_max_offset=8.0,
                 use_sat_global_se2=False,
                 sat_se2_max_translation=8.0,
                 sat_se2_max_yaw=15.0,
                 use_sat_bev_refine=False,   # ✅ 新增开关
                 bev_fusion_mode='dynamic_aware',
                 export_vis=False,           # ✅ 图4/图5：导出中间量到 self._vis
                 **kwargs):
        super(DualViewTransformerFull_SAT, self).__init__(**kwargs)
        self.loss_semantic_weight = loss_semantic_weight
        self.depth_threshold = depth_threshold / self.D
        self.semantic_threshold = semantic_threshold
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_height = num_height
        self.collapse_z = collapse_z
        self.sat = if_sat
        self.hiddenC = 256
        self.count_ = 0
        self.Z = 16
        self.export_vis = export_vis   # ✅ 图4/图5 中间量采集开关
        valid_fusion_modes = {
            'addition', 'concatenation', 'factorization',
            'generic_attention', 'dynamic_aware'
        }
        if bev_fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f'Unsupported bev_fusion_mode={bev_fusion_mode!r}; '
                f'expected one of {sorted(valid_fusion_modes)}')
        self.bev_fusion_mode = bev_fusion_mode

        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                            self.out_channels, self.D+2, **depthnet_cfg)
        self.fuser = DualFeatFusion(2*(self.out_channels), self.out_channels)
        self.geom_att = BEVGeomAttention()

         # ✅ 可变形对齐模块（放在所有 SAT head 之前）
        self.use_sat_deform = use_sat_deform
        if use_sat_deform:
            # sat_feat 在 conv_last 之前是 UNet18 输出（128ch）
            # bev_feat 是 fuser 输出（out_channels=80ch）
            if use_sat_global_se2:
                meters_per_cell = ((pc_range[3] - pc_range[0]) /
                                   float(self.bev_w))
                self.sat_deform_align = SATSE2DeformAlign(
                    cam_ch=self.out_channels,
                    sat_ch=128,
                    deform_groups=sat_deform_groups,
                    local_max_offset=sat_deform_max_offset,
                    max_translation_m=sat_se2_max_translation,
                    max_yaw_deg=sat_se2_max_yaw,
                    meters_per_cell=meters_per_cell,
                )
            else:
                self.sat_deform_align = SATDeformAlign(
                    cam_ch=self.out_channels,   # 80
                    sat_ch=128,                 # UNet18 输出通道
                    deform_groups=sat_deform_groups,
                    max_offset=sat_deform_max_offset,
                )

        # ✅ 添加 ProbNet 初始化
        # self.prob = ProbNet(
        #     in_channels=self.out_channels,
        #     scale_factor=1,
        #     with_centerness=False,
        #     loss_weight=0.2,
        #     bev_size=(self.bev_h, self.bev_w)
        # )

        # self.dynamic_refine = nn.Sequential(
            # nn.Conv2d(2, 1, kernel_size=1, padding=0)
        # )

        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=3, padding=1),
        # )
        # _
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=1),
        # )
        # 2
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=3, padding=1),
        # )
        # 3
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=1),
        # )
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=3, padding=1),
        # )
        self.dynamic_refine = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
        # )
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=3, padding=1),
        # )
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 1, kernel_size=1),
        # )
        # self.dynamic_refine = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 1, kernel_size=3, padding=1),
        # )

        # self.positional_encoding = LearnedPositionalEncoding(
        #     1, self.bev_h, self.bev_w
        # )
        self.down = nn.Conv2d(self.out_channels*2, self.out_channels, kernel_size=3, padding=1)
        # self.down = nn.Sequential(nn.Conv2d(self.out_channels*2, self.out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1))

        # self.align = Align(80)
        self.conv_last = nn.Sequential(nn.Conv2d(128, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.conv_last_sem = nn.Conv2d(128, 7, kernel_size=3, padding=1)
        self.conv_last_height = nn.Conv2d(128, 17, kernel_size=3, padding=1)
        self.skip_aux = False   # ✅ 推理模式：跳过训练辅助 head（sat_sem/sat_height，仅训练 loss 用）

        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=1.0)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

        self.use_sat_bev_refine = use_sat_bev_refine   # ✅ 补上这一行
        if use_sat_bev_refine:
            self.sat_bev_refine = nn.Sequential(
                nn.Conv2d(self.out_channels * 2, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
            )
            # ✅ 零初始化：起步残差=0，不改变原 projected_sem_logit
            nn.init.zeros_(self.sat_bev_refine[-1].weight)
            nn.init.zeros_(self.sat_bev_refine[-1].bias)
        else:
            self.sat_bev_refine = None

    def get_reference_points_3d(self, H, W, Z=8, num_points_in_pillar=13, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in HT.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in HT, has \
                shape (bs, D, HW, 3).
        """
        zs = torch.linspace(1/(2*num_points_in_pillar), 1-1/(2*num_points_in_pillar), num_points_in_pillar, dtype=dtype,device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W)

        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    @force_fp32()
    def get_projection(self, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda):
        B, N, _, _ = sensor2egos.shape
        rots = sensor2egos[:, :, :3, :3]
        trans = sensor2egos[:, :, :3, 3]
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        inv_sensor2ego = torch.inverse(rots.cpu()).cuda()
        # print(rots.shape, intrins.shape, inv_sensor2ego.shape, torch.inverse(bda).shape)
        lidar2img_R = intrins.matmul(inv_sensor2ego).matmul(torch.inverse(bda.cpu()).cuda())
        lidar2img_t = -intrins.matmul(inv_sensor2ego).matmul(trans.unsqueeze(-1))
        lidar2img = torch.cat((lidar2img_R, lidar2img_t), -1)
        img_aug = torch.cat((post_rots, post_trans.unsqueeze(-1)), -1)
        return lidar2img, img_aug

    @force_fp32()
    def get_sampling_point(self, reference_points, pc_range, depth_range, lidar2img, img_aug, image_shapes):
        # B, bev_z, bev_h* bev_w, 3
        reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

        # B, D, HW, 3
        B, Z, num_query = reference_points.size()[:3]
        reference_points = reference_points.reshape(B, -1 , 3)
        num_cam = lidar2img.size(1)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.view(B, 1, Z*num_query, 4).repeat(1, num_cam, 1, 1)

        lidar2img = lidar2img.view(B, num_cam, 1, 3, 4)
        img_aug = img_aug.view(B, num_cam, 1, 3, 4)

        reference_points = lidar2img.matmul(reference_points.unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        referenece_depth = reference_points[..., 2:3].clone()
        bev_mask = (reference_points[..., 2:3] > eps)

        reference_points_cam = torch.cat((reference_points[..., 0:2] / torch.maximum(
            reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])*eps), reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])), -1)

        reference_points_cam = torch.matmul(img_aug,
                                        reference_points_cam.unsqueeze(-1)).squeeze(-1)

        reference_points_cam = reference_points_cam[..., 0:2]
        reference_points_cam[..., 0] /= image_shapes[1]
        reference_points_cam[..., 1] /= image_shapes[0]

        reference_points_cam= reference_points_cam.view(B,num_cam,Z,num_query,2)
        referenece_depth= referenece_depth.view(B,num_cam,Z,num_query,1)
        bev_mask= bev_mask.view(B,num_cam,Z,num_query,1)


        bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > 0.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0))
        # D, B, N, num_query, 1
        if depth_range is not None:
            referenece_depth = (referenece_depth-depth_range[0])/(depth_range[1]-depth_range[0])
            bev_mask = (bev_mask & (referenece_depth > 0.0)
                        & (referenece_depth < 1.0))
        # [B, N, Z*Nq, 1] bev_mask
        # [B, N, Z*Nq, 2] reference_points_cam
        # [B, N, Z*Nq, 1] referenece_depth
        bev_mask = torch.nan_to_num(bev_mask)
        return torch.cat((reference_points_cam,referenece_depth),-1), bev_mask

    def init_acceleration_ht(self, coor, mask):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)

        self.ranks_bev_ht = ranks_bev.int().contiguous()
        self.ranks_feat_ht = ranks_feat.int().contiguous()
        self.ranks_depth_ht = ranks_depth.int().contiguous()
        self.interval_starts_ht = interval_starts.int().contiguous()
        self.interval_lengths_ht = interval_lengths.int().contiguous()

    def fast_sampling(self, coor, mask, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                1, int(self.bev_h), int(self.bev_w)
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def fast_sample_prepare(self, coor, mask):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the image space in
                shape (B, N, ZNq 3).
            mask (torch.tensor): mask of points in the imaage space in
                shape (B, N, ZNq, 1).
        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, Z, Nq, _ = coor.shape
        num_points = B * N * Z * Nq
        # record the index of selected points for acceleration purpose
        ranks_bev = torch.range(
            0, num_points // (N*Z) - 1, dtype=torch.int, device=coor.device)
        ranks_bev = ranks_bev.reshape(B, 1, 1, Nq)
        ranks_bev = ranks_bev.expand(B, N, Z, Nq).flatten()
        # convert coordinate into the image feat space
        coor[..., 0] *= self.W
        coor[..., 1] *= self.H
        coor[..., 2] *= self.D
        # [B, N, Z, Nq, 3]
        coor = coor.round().long().view(num_points, 3)
        coor[..., 0].clamp_(min=0, max=self.W-1)
        coor[..., 1].clamp_(min=0, max=self.H-1)
        coor[..., 2].clamp_(min=0, max=self.D-1)
        batch_idx = torch.range(0, B*N-1).reshape(B*N, 1). \
            expand(B*N, num_points // (B*N)).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = mask.reshape(-1)
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_bev = \
            coor[kept], ranks_bev[kept]

        ranks_depth = coor[:, 3] * (self.D * self.W * self.H)
        ranks_depth += coor[:, 2] * (self.W * self.H)
        ranks_depth += coor[:, 1] * self.W + coor[:, 0]
        depth_size = B * N * self.D * self.W * self.H
        ranks_depth.clamp_(min=0, max=depth_size-1)

        ranks_feat = coor[:, 3] * (self.W * self.H)
        ranks_feat += coor[:, 1] * self.W + coor[:, 0]
        feat_size = B * N * self.W * self.H

        ranks_feat.clamp_(min=0, max=feat_size-1)

        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.lidar2img, self.img_aug = self.get_projection(*input[1:7])
            self.init_acceleration_v2(coor)

            self.W = self.input_size[1] / self.downsample
            self.H = self.input_size[0] / self.downsample
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, num_points_in_pillar=self.num_height, bs=1)
            coor, mask = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], self.lidar2img, self.img_aug, self.input_size)
            self.init_acceleration_ht(coor, mask)

            self.initial_flag = False

    def get_lss_bev_feat(self, input, depth, tran_feat, kept=None):
        """
        LSS投影（特征+语义一起投影）
        Args:
            tran_feat: (B*N, C+2, H, W) 特征 + 2个语义类别
        Returns:
            bev_feat: (B, C, bev_h, bev_w) 特征
            bev_sem: (B, 2, bev_h, bev_w) 语义
        """
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, tran_feat.shape[1], H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])
            bev_feat_with_sem = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)
            bev_feat_with_sem = bev_feat_with_sem.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])
            bev_feat_with_sem = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, tran_feat.shape[1], H, W))

        # 分离特征和语义
        bev_feat = bev_feat_with_sem[:, :-1, :, :]  # (B, C, bev_h, bev_w)
        bev_sem = bev_feat_with_sem[:, -1:, :, :]   # (B, 2, bev_h, bev_w)

        return bev_feat, bev_sem

    def get_ht_bev_feat(self, input, depth, tran_feat, bev_mask=None):
        """
        HT投影（特征+语义一起投影）
        Args:
            tran_feat: (B*N, C+2, H, W) 特征 + 2个语义类别
        Returns:
            bev_feat: (B, C, bev_h, bev_w) 特征
            bev_sem: (B, 2, bev_h, bev_w) 语义
        """
        B, N, C, H, W = input[0].shape
        self.H = H
        self.W = W

        # Prob-Sampling
        if self.accelerate:
            feat = tran_feat.view(B, N, tran_feat.shape[1], H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])
            bev_feat_with_sem = bev_pool_v2(depth, feat, self.ranks_depth_ht,
                                   self.ranks_feat_ht, self.ranks_bev_ht,
                                   bev_feat_shape, self.interval_starts_ht,
                                   self.interval_lengths_ht)
            if bev_mask is not None:
                bev_feat_with_sem = bev_feat_with_sem * bev_mask
            bev_feat_with_sem = bev_feat_with_sem.squeeze(2)
        else:
            lidar2img, img_aug = self.get_projection(*input[1:7])
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, bs=B, num_points_in_pillar=self.num_height)
            coor, mask = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], lidar2img, img_aug, self.input_size)
            if bev_mask is not None:
                mask = bev_mask * mask.view(B, N, self.num_height, self.bev_h, self.bev_w)
            bev_feat_with_sem = self.fast_sampling(
                coor, mask, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, tran_feat.shape[1], H, W))

        # 分离特征和语义
        bev_feat = bev_feat_with_sem[:, :-1, :, :]  # (B, C, bev_h, bev_w)
        bev_sem = bev_feat_with_sem[:, -1:, :, :]   # (B, 2, bev_h, bev_w)

        return bev_feat, bev_sem

    def view_transform(self, input, depth, tran_feat):
        """
        投影特征和语义
        Args:
            tran_feat: (B*N, C+2, fH, fW) 特征 + 2个语义类别
        Returns:
            lss_feat, ht_feat: BEV特征（C通道）
            lss_sem, ht_sem: 投影后的语义（2通道）
        """
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape

        # print('tran_feat: ', tran_feat.shape)
        lss_feat, lss_sem = self.get_lss_bev_feat(input, depth, tran_feat)
        ht_feat, ht_sem = self.get_ht_bev_feat(input, depth, tran_feat)

        return lss_feat, ht_feat, lss_sem, ht_sem, depth


    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        # remove point not in depth range
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
        # gt_semantic_depths = gt_depths * gt_semantics

        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.downsample * self.downsample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.downsample,
                                   W // self.downsample)
        # gt_semantics = F.one_hot(gt_semantics.long(),
        #                       num_classes=2).view(-1, 2).float()
        gt_semantics = gt_semantics.view(-1).long()
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths = (gt_depths -
                     (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        # print(gt_depths.shape)
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(
                                  -1, self.D + 1)[:, 1:].float()

        return gt_depths, gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, depth_labels, depth_preds, semantic_labels, semantic_preds):
        """
        深度和语义的监督损失
        Args:
            depth_labels: (B*N*fH*fW, D) one-hot depth
            depth_preds: (B*N, D, fH, fW)
            semantic_labels: (B*N*fH*fW,) 语义标签 (0=static, 1=dynamic)
            semantic_preds: (B*N, 2, fH, fW)
        """
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)

        # depth mask
        depth_mask = torch.max(depth_labels, dim=1).values > 0.0

        depth_labels = torch.argmax(depth_labels, dim=1)
        depth_labels = depth_labels[depth_mask]
        depth_preds = depth_preds[depth_mask]
        semantic_labels = semantic_labels[depth_mask]
        semantic_preds = semantic_preds[depth_mask]

        with autocast(enabled=False):
            depth_loss = F.cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, depth_preds.size(0))

            # Focal Loss for semantic
            alpha = 0.25
            gamma = 2
            ce_loss = F.cross_entropy(semantic_preds, semantic_labels, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = alpha * (1 - pt).pow(gamma)
            semantic_loss = ce_loss * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))

        return self.loss_depth_weight * depth_loss, self.loss_semantic_weight * semantic_loss

    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        if self.sat:
            sat_feat = input[8]


        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        x = self.depth_net(x, mlp_input, stereo_metas)

        depth = x[:, :self.D, ...]
        semantic = x[:, self.D:self.D + 2, ...]
        tran_feat = x[:, self.D + 2:self.D + 2 + self.out_channels, ...]

        # 拼接后一起投影
        tran_feat_with_sem = torch.cat([tran_feat, semantic[:, 1:2, :, :]], dim=1)  # (B*N, C+1, fH, fW)
        filter_depth = torch.where(depth < self.depth_threshold, torch.zeros_like(depth), depth)

        # 投影特征和语义
        lss_feat, ht_feat, lss_sem, ht_sem, filter_depth = self.view_transform(
            input, filter_depth, tran_feat_with_sem)

        # print(f"[DEBUG] lss_sem.shape: {lss_sem.shape}, ht_sem.shape: {ht_sem.shape}")
        # 期望: lss_sem.shape: torch.Size([B, 1, 200, 200])

        bev_feat = self.fuser(ht_feat, lss_feat)  # (B, C, bev_h, bev_w)
        # bev_feat = self.fuser(lss_feat, ht_feat)  # (B, C, bev_h, bev_w)
        # bev_feat = lss_feat

        if self.use_sat_deform:
            sat_feat = self.sat_deform_align(bev_feat, sat_feat)

        # sat_feat = self.align(sat_feat, bev_feat)
        # The satellite auxiliary heads are intentionally skipped in normal
        # inference: they are not on the occupancy prediction path.  For the
        # qualitative height-guided projection only, opt in explicitly so the
        # benchmark latency remains unchanged.
        export_sat_height = (
            self.export_vis and os.environ.get('OCC_VIS_EXPORT_SAT_HEIGHT') == '1'
        )
        if (self.training and not self.skip_aux) or export_sat_height:
            sat_depth = self.conv_last_height(sat_feat)
            sat_sem = self.conv_last_sem(sat_feat)
        else:
            # 推理模式：跳过训练辅助 head（输出仅用于训练 loss，不参与推理）
            sat_depth = None
            sat_sem = None
        sat_feat_processed = self.conv_last(sat_feat)

        # dtype = input[0].dtype
        # mask = torch.zeros((B, self.bev_h, self.bev_w), device=ht_feat.device).to(dtype)
        # bev_pos = self.positional_encoding(mask).to(dtype)
        # print('111111111111: ', bev_pos.shape)

        projected_sem_logit = self.dynamic_refine(torch.cat([
            lss_sem,
            ht_sem,
        ], dim=1))

        # ✅ SAT+BEV 联合调节动态 logit：
        # sat 知道静态结构（道路/建筑），bev 知道动态物体位置
        # 二者 concat 学一个残差修正 logit，弥补纯投影语义的类别偏差
        if self.use_sat_bev_refine and self.sat_bev_refine is not None:
            projected_sem_logit = projected_sem_logit + self.sat_bev_refine(
                torch.cat([sat_feat_processed.detach(), bev_feat.detach()], dim=1)
            )

        # Cross-view BEV ablations.  Every branch receives the same aligned
        # camera/satellite tensors; only the fusion operator changes.
        mpdm_dynamic_prob = torch.sigmoid(projected_sem_logit)
        mpdm_static_gate = 1 - mpdm_dynamic_prob
        bev_feat_pre_psdf = None
        if self.bev_fusion_mode == 'addition':
            bev_feat_final = bev_feat + sat_feat_processed
            bev_feat_pre_psdf = bev_feat_final
            psdf_attention = torch.ones_like(projected_sem_logit)
        elif self.bev_fusion_mode == 'concatenation':
            bev_feat_final = self.down(
                torch.cat((bev_feat, sat_feat_processed), dim=1))
            bev_feat_pre_psdf = bev_feat_final
            psdf_attention = torch.ones_like(projected_sem_logit)
        else:
            # Static--dynamic factorization: satellite features are admitted
            # only where the current-view MPDM predicts a static region.
            bev_feat_final = self.down(torch.cat(
                (bev_feat, sat_feat_processed * mpdm_static_gate), dim=1))
            bev_feat_pre_psdf = bev_feat_final
            if self.bev_fusion_mode == 'factorization':
                psdf_attention = torch.ones_like(projected_sem_logit)
            elif self.bev_fusion_mode == 'generic_attention':
                # Spatial attention without the dynamic reliability cue.
                psdf_attention = self.geom_att(
                    bev_feat_final, torch.zeros_like(projected_sem_logit))
                bev_feat_final = psdf_attention * bev_feat_final
            else:  # dynamic_aware
                psdf_attention = self.geom_att(
                    bev_feat_final, projected_sem_logit)
                bev_feat_final = psdf_attention * bev_feat_final

        # ✅ 图4/图5：采集中间量（CPU，供 vis_paper.py 可视化）
        if self.export_vis:
            self._vis = dict(
                perspective_sem=semantic.detach().float().cpu(),      # (B*N, 2, fH, fW) 透视可移动概率
                lss_sem=lss_sem.detach().float().cpu(),               # (B, 1, 200, 200) LSS 投影 mask
                ht_sem=ht_sem.detach().float().cpu(),                 # (B, 1, 200, 200) Uni-SA 投影 mask
                lss_feat=lss_feat.detach().float().cpu(),             # (B, C, 200, 200) LSS BEV
                ht_feat=ht_feat.detach().float().cpu(),               # (B, C, 200, 200) HT BEV
                proj_sem_logit=projected_sem_logit.detach().float().cpu(),   # (B, 1, 200, 200) refined MPDM
                mpdm_dynamic_prob=mpdm_dynamic_prob.detach().float().cpu(), # (B, 1, 200, 200)
                mpdm_static_gate=mpdm_static_gate.detach().float().cpu(),   # (B, 1, 200, 200)
                bev_feat_pre_psdf=bev_feat_pre_psdf.detach().float().cpu(), # before PSDF attention
                sat_feat_processed=sat_feat_processed.detach().float().cpu(), # (B, 80, 200, 200)
                bev_feat=bev_feat.detach().float().cpu(),             # (B, 80, 200, 200) fuser 后 BEV
                bev_feat_final=bev_feat_final.detach().float().cpu(), # (B, 80, 200, 200) 最终 BEV
            )
            if sat_depth is not None:
                # B x Z x Dy x Dx logits; exported only for the offline
                # GT+satellite-height-guided BEV visualisation.
                self._vis['sat_height'] = sat_depth.detach().float().cpu()

        return bev_feat_final, depth, "bev_mask_logit", semantic, sat_sem, sat_depth, projected_sem_logit, sat_feat_processed, bev_feat

    def visualize_comparison(self, lss_sem, ht_sem, projected_sem_logit, bev_mask_prob, step,
                             save_dir=None):
        """可视化对比：LSS动态 | HT动态 | 投影语义 | ProbNet"""
        if save_dir is None:
            return
        import os

        os.makedirs(save_dir, exist_ok=True)

        def to_heatmap(tensor):
            map_np = tensor[0, 0].detach().cpu().numpy()
            map_uint8 = (map_np * 255).astype(np.uint8)
            return cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET)

        # 使用动态通道（索引1）
        lss_colored = to_heatmap(torch.sigmoid(lss_sem[:, 1:2, :, :]))
        ht_colored = to_heatmap(torch.sigmoid(ht_sem[:, 1:2, :, :]))
        proj_colored = to_heatmap(torch.sigmoid(projected_sem_logit))
        prob_colored = to_heatmap(bev_mask_prob)

        comparison = np.hstack([lss_colored, ht_colored, proj_colored, prob_colored])
        cv2.imwrite(os.path.join(save_dir, f'comparison_{step:06d}.png'), comparison)

    @force_fp32()
    def get_projected_sem_loss(self, gt_bev_mask, proj_dynamic):
        """
        投影语义的辅助监督（对齐 get_bev_mask_loss）
        Args:
            gt_bev_mask: (B, bev_h, bev_w) GT动态掩码
            projected_sem_logit: (B, 1, bev_h, bev_w) 动态 logit
        """
        B, bev_h, bev_w = gt_bev_mask.shape

        # 展平
        b = gt_bev_mask.reshape(B, bev_w * bev_h).permute(1, 0).to(torch.float)
        a = proj_dynamic.reshape(B, bev_w * bev_h).permute(1, 0)

        with autocast(enabled=False):
            # BCE loss
            proj_ce_loss = self.ce_loss(a, b)

            # Dice loss
            proj_dice_loss = self.dice_loss(
                proj_dynamic.reshape(B, -1),
                gt_bev_mask.reshape(B, -1)
            )

        return dict(
            proj_ce_loss=0.2 * proj_ce_loss,
            proj_dice_loss=0.2 * proj_dice_loss
        )

    def get_loss(self, depth, semantic, gt_depth, gt_semantic):
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth, loss_ce_semantic = \
            self.get_depth_and_semantic_loss(depth_labels, depth, semantic_labels, semantic)
        return loss_depth, loss_ce_semantic

    @force_fp32()
    def get_sat_depth_loss(self, depth_labels, depth_preds, valid_mask):
        """
        Args:
            depth_labels: (B, 1, img_h, img_w)
            depth_preds: (B, Dz, fH, fW)
            valid_mask: (B, img_h, img_w)
        Returns:

        """

        depth_preds = depth_preds.permute(0, 2, 3,
                                    1).contiguous().view(-1, self.Z+1)[valid_mask]

        with autocast(enabled=False):

            sat_depth_loss = F.cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, depth_preds.size(0))
        return self.loss_depth_weight * sat_depth_loss

    @force_fp32()
    def get_sat_sem_loss(self, sem_labels, sem_preds, valid_mask):
        """
        Args:
            sem_labels: (B, 7, img_h, img_w)
            sem_preds: (B, 7, fH, fW)
        Returns:

        """

        sem_preds = sem_preds.permute(0, 2, 3, 1).contiguous().view(-1, 7)[valid_mask]

        alpha = 0.25
        gamma = 2

        with autocast(enabled=False):

            ce_loss = F.cross_entropy(sem_preds, sem_labels.long(), reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = alpha * (1 - pt).pow(gamma)
            sat_sem_loss = ce_loss * focal_weight
            sat_sem_loss = sat_sem_loss.sum() / max(1, len(sat_sem_loss))

        return sat_sem_loss * 0.5


@NECKS.register_module()
class DualViewTransformerStereo_SAT(DualViewTransformerFull_SAT):
    def __init__(self,  **kwargs):
        super(DualViewTransformerStereo_SAT, self).__init__(**kwargs)
        # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)
