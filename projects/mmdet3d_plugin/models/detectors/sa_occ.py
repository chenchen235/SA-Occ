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

@DETECTORS.register_module()
class SA_OCC(DualBEV):
    def __init__(self, *args, upsample=False, ifsat=True, sat_img_backbone=None, occ_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sat = ifsat
        if ifsat:
            self.sat_img_backbone = builder.build_backbone(sat_img_backbone)
        else:
            self.sat_img_backbone = None

        self.occ_head = builder.build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        
        self.count = 0

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
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape

        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)
        if self.sat:

            x_sat = self.sat_img_backbone(img_sat)

        else:
            x_sat = None

        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            # print(len(x))
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat, x_sat, img_sat

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
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        if self.sat:
            return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda, img_sat]
        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]
    
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img[1:7]
        mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)
        if self.sat:
            x, _, x_sat, img_sat = self.image_encoder(img[0], img[7])
            inputs=[x, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input, x_sat]#, sat_sem, sat_height]
        else:
            x, _, _ = self.image_encoder(img[0], "_")
            inputs=[x, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input]

        x, depth, bev_mask, semantic, sat_sem, sat_height = self.img_view_transformer(inputs)
        # bev_sat = self.satbev_encoder(bev_sat)
        x = self.bev_encoder(x)
        return [x], depth, bev_mask, semantic, sat_sem, sat_height, img_sat #, sat_depth, sat_image#, [bev_sat], sat_image
    
    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        # img_feats, depth, bev_mask, bev_sats, sat_image = self.extract_img_feat(img, img_metas, **kwargs)
        img_feats, depth, bev_mask, semantic, sem_sat, height_sat, img_sat = self.extract_img_feat(img, img_metas, **kwargs)

        pts_feats = None
        return (img_feats, pts_feats, depth, bev_mask, semantic, sem_sat, height_sat, img_sat)#, sat_image)#, sat_depth, sat_image)#, bev_sats, sat_image)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        img_feats, pts_feats, depth, bev_mask, semantic, sem_sat, height_sat, img_sat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        loss_depth, loss_ce_semantic = \
            self.img_view_transformer.get_loss(depth, semantic, gt_depth, gt_semantic)
       
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_bev_mask = (voxel_semantics >= 0) & (voxel_semantics <= 10)
        gt_bev_mask = gt_bev_mask.any(dim=-1).permute(0, 2, 1)
        gt_bev_mask = gt_bev_mask | kwargs['gt_bev_mask']

        position_encoding = torch.arange(voxel_semantics.size(3), dtype=torch.long).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(voxel_semantics.device)
        position_encoding = position_encoding.expand_as(voxel_semantics)
        categories = torch.arange(11, 17, dtype=torch.long).to(voxel_semantics.device)

        valid_category_mask = torch.stack([voxel_semantics == category for category in categories], dim=0).any(dim=0).to(voxel_semantics.device)

        valid_mask = torch.any(valid_category_mask, dim=3).permute(0, 2, 1)#.reshape(-1)
        position_encoded_mask = position_encoding * valid_category_mask
        dz_max_indices = torch.argmax(position_encoded_mask, dim=-1).to(voxel_semantics.device).permute(0, 2, 1) # 200 x 200 11-16 6 cate 有些是valid类别但是高度是0，其他的高度为0表示无意义
        
        sat_semantic_gt = voxel_semantics.permute(0, 2, 1, 3).gather(3, dz_max_indices.unsqueeze(-1).long()).squeeze(-1)
        sat_semantic_gt = torch.where(valid_mask, sat_semantic_gt-10, torch.zeros_like(sat_semantic_gt))
        
        sat_semantic_gt = sat_semantic_gt.view(-1)[valid_mask.reshape(-1)]
        sat_dz_gt = dz_max_indices.reshape(-1)[valid_mask.reshape(-1)]
        
        loss_sat_height = self.img_view_transformer.get_sat_depth_loss(sat_dz_gt, height_sat, valid_mask.reshape(-1))
        loss_sat_sem = self.img_view_transformer.get_sat_sem_loss(sat_semantic_gt, sem_sat, valid_mask.reshape(-1))
        
        losses = dict(loss_depth=loss_depth, loss_ce_semantic=loss_ce_semantic, loss_sat_height=loss_sat_height, loss_sat_sem=loss_sat_sem)

        loss_bev_mask = self.img_view_transformer.prob.get_bev_mask_loss(gt_bev_mask, bev_mask)
        losses.update(loss_bev_mask)
        
        occ_bev_feature = img_feats[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2, mode='bilinear', align_corners=True)

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
    
    def forward_satocc_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.satocc_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_satocc = self.satocc_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_satocc

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _, _, _, sem_sat, height_sat, img_sat = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            bev_sat = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

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

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _, _, _, _, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        
        return outs