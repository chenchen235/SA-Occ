
_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True

plugin_dir = 'projects/mmdet3d_plugin/'
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40., -40., -3.0, 40., 40., 3.4]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='SA_OCC_Stereo4D',
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    sat_img_backbone = dict(
        type='ResNetUNet18'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='DualViewTransformerStereo_SAT',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        depthnet_cfg=dict(use_dcn=False,
                          stereo=True,
                          bias=5.),
        loss_depth_weight=0.05,
        downsample=16),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=256,
        out_dim=256,    # out_dim=128 for M0!!!
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
    )
)

# Data
dataset_type = 'NuScenesDatasetSATOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        load_point_label=True,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='PrepareSATImageInputs', sequential=True),
    dict(type='GetBEVMask', point_cloud_range=point_cloud_range, voxel_size=[0.4,0.4,6.4], downsample_ratio=1.),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_mask', 'gt_depth', 'gt_semantic', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='PrepareSATImageInputs', sequential=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from = "/workspace/FlashOCC/dualbev4d-r50-cbgs.pth"


evaluation = dict(interval=1, start=0, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# 4d
# ===> others - IoU = 10.37
# ===> barrier - IoU = 46.88
# ===> bicycle - IoU = 21.43
# ===> bus - IoU = 45.76
# ===> car - IoU = 51.72
# ===> construction_vehicle - IoU = 21.51
# ===> motorcycle - IoU = 22.78
# ===> pedestrian - IoU = 23.53
# ===> traffic_cone - IoU = 22.54
# ===> trailer - IoU = 34.37
# ===> truck - IoU = 38.79
# ===> driveable_surface - IoU = 83.25
# ===> other_flat - IoU = 43.62
# ===> sidewalk - IoU = 54.61
# ===> terrain - IoU = 58.97
# ===> manmade - IoU = 47.41
# ===> vegetation - IoU = 42.48
# ===> mIoU of 6019 samples: 39.41
# {'mIoU': array([0.104, 0.469, 0.214, 0.458, 0.517, 0.215, 0.228, 0.235, 0.225,
#        0.344, 0.388, 0.833, 0.436, 0.546, 0.59 , 0.474, 0.425, 0.909])}

# ===> per class IoU of 6019 samples:
# ===> others - IoU = 11.58
# ===> barrier - IoU = 48.33
# ===> bicycle - IoU = 21.25
# ===> bus - IoU = 45.38
# ===> car - IoU = 53.01
# ===> construction_vehicle - IoU = 23.78
# ===> motorcycle - IoU = 23.6
# ===> pedestrian - IoU = 23.93
# ===> traffic_cone - IoU = 22.96
# ===> trailer - IoU = 35.37
# ===> truck - IoU = 40.24
# ===> driveable_surface - IoU = 83.75
# ===> other_flat - IoU = 45.24
# ===> sidewalk - IoU = 55.71
# ===> terrain - IoU = 60.11
# ===> manmade - IoU = 51.08
# ===> vegetation - IoU = 45.26
# ===> mIoU of 6019 samples: 40.62
# {'mIoU': array([0.116, 0.483, 0.213, 0.454, 0.53 , 0.238, 0.236, 0.239, 0.23 ,
#        0.354, 0.402, 0.838, 0.452, 0.557, 0.601, 0.511, 0.453, 0.915])}

# ===> others - IoU = 10.89
# ===> barrier - IoU = 48.45
# ===> bicycle - IoU = 23.48
# ===> bus - IoU = 45.78
# ===> car - IoU = 52.77
# ===> construction_vehicle - IoU = 24.49
# ===> motorcycle - IoU = 23.61
# ===> pedestrian - IoU = 24.23
# ===> traffic_cone - IoU = 22.72
# ===> trailer - IoU = 34.96
# ===> truck - IoU = 40.28
# ===> driveable_surface - IoU = 83.49
# ===> other_flat - IoU = 44.09
# ===> sidewalk - IoU = 55.54
# ===> terrain - IoU = 59.88
# ===> manmade - IoU = 51.22
# ===> vegetation - IoU = 45.25
# ===> mIoU of 6019 samples: 40.65
# {'mIoU': array([0.109, 0.485, 0.235, 0.458, 0.528, 0.245, 0.236, 0.242, 0.227,
#        0.35 , 0.403, 0.835, 0.441, 0.555, 0.599, 0.512, 0.453, 0.915])}

# noalign
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.84
# ===> barrier - IoU = 48.19
# ===> bicycle - IoU = 22.11
# ===> bus - IoU = 45.58
# ===> car - IoU = 52.82
# ===> construction_vehicle - IoU = 24.41
# ===> motorcycle - IoU = 23.54
# ===> pedestrian - IoU = 23.77
# ===> traffic_cone - IoU = 22.89
# ===> trailer - IoU = 35.82
# ===> truck - IoU = 39.97
# ===> driveable_surface - IoU = 83.55
# ===> other_flat - IoU = 44.42
# ===> sidewalk - IoU = 55.19
# ===> terrain - IoU = 59.59
# ===> manmade - IoU = 51.13
# ===> vegetation - IoU = 45.26
# ===> mIoU of 6019 samples: 40.53
# {'mIoU': array([0.108, 0.482, 0.221, 0.456, 0.528, 0.244, 0.235, 0.238, 0.229,
#        0.358, 0.4  , 0.835, 0.444, 0.552, 0.596, 0.511, 0.453, 0.915])}