import torch
import torch.nn as nn
import torchvision.models as models
from mmdet3d.models import BACKBONES
from ..model_utils.depthnet import ASPP
import torch.nn.functional as F


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ResNetUNet18_Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        base_model = models.resnet18(pretrained=True) # 64 64 128 256

        self.base_layers = list(base_model.children())

        self.aspp = ASPP(inplanes=256, mid_channels=256, outplanes=256, dilations = [1, 3, 6, 9]) # 1248 1369
        self.gate_layer0 = nn.Sequential(*self.base_layers[:2])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        layer0 = self.layer0(input)
        layer0 = self.sigmoid(self.gate_layer0(input)) * layer0

        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = self.aspp(layer3)
        return layer0, layer1, layer2, layer3

@BACKBONES.register_module()
class ResNetUNet18(nn.Module):

    def __init__(self, pretrained_path=None):
        super().__init__()

        self.encoder =ResNetUNet18_Encoder() # 64 64 128 256
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path)
            self.encoder.load_state_dict(ckpt, strict=True)
            print("ur 18 load from", pretrained_path)

        self.layer1_1x1 = convbnrelu(64, 128, 3, 1)
        self.layer0_1x1 = convbnrelu(64, 128, 3, 1)
        self.conv_up2 = convbnrelu(384, 128, 3, 1)
        self.conv_up1 = convbnrelu(256, 128, 3, 1)
        self.conv_up0 = convbnrelu(256, 128, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input, return_multiscale=False):

        layer0, layer1, layer2, layer3 = self.encoder(input)

        # ✅ 用 interpolate 对齐 skip 尺寸, 替代固定 2x upsample, 支持非 16 倍数输入(如 200×200)
        x = F.interpolate(layer3, size=layer2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer2], dim=1)
        d2 = self.conv_up2(x)                    # decoder: 128 @ 1/8

        x = F.interpolate(d2, size=layer1.shape[-2:], mode='bilinear', align_corners=True)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        d1 = self.conv_up1(x)                    # decoder: 128 @ 1/4

        x = F.interpolate(d1, size=layer0.shape[-2:], mode='bilinear', align_corners=True)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        d0 = self.conv_up0(x)                    # decoder: 128 @ 1/2

        if return_multiscale:
            return d0, [d0, d1]    # ✅ 去掉 d25，只返回 d0@200 和 d1@100
        return d0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet18()
    model = model.to(device)

    summary(model, input_size=(3, 400, 400))

# ===> per class IoU of 6019 samples: 328s / 原330
# ===> others - IoU = 10.17
# ===> barrier - IoU = 46.03
# ===> bicycle - IoU = 20.94
# ===> bus - IoU = 46.31
# ===> car - IoU = 51.27
# ===> construction_vehicle - IoU = 22.27
# ===> motorcycle - IoU = 22.44
# ===> pedestrian - IoU = 23.57
# ===> traffic_cone - IoU = 21.62
# ===> trailer - IoU = 33.63
# ===> truck - IoU = 38.28
# ===> driveable_surface - IoU = 82.87
# ===> other_flat - IoU = 43.6
# ===> sidewalk - IoU = 54.84
# ===> terrain - IoU = 58.44
# ===> manmade - IoU = 46.88
# ===> vegetation - IoU = 41.66
# ===> mIoU of 6019 samples: 39.11 / 39.03
# {'mIoU': array([0.102, 0.46 , 0.209, 0.463, 0.513, 0.223, 0.224, 0.236, 0.216,
#        0.336, 0.383, 0.829, 0.436, 0.548, 0.584, 0.469, 0.417, 0.902])}
