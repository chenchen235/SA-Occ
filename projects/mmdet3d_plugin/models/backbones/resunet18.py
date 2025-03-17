import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from mmdet3d.models import BACKBONES
from ..model_utils.depthnet import ASPP



def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

@BACKBONES.register_module()
class ResNetUNet18(nn.Module):

    def __init__(self):
        super().__init__()

        base_model = models.resnet18(pretrained=True)

        self.base_layers = list(base_model.children())

        self.aspp = ASPP(inplanes=128, mid_channels=256, outplanes=256)
        self.gate_layer0 = nn.Sequential(*self.base_layers[:2])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)

        self.layer0_1x1 = convrelu(64, 256, 3, 1)
        # self.gate_layer1 =  nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64))
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)

        self.layer1_1x1 = convrelu(64, 256, 3, 1)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        # self.layer2_1x1 = convrelu(128, 256, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up1 = convrelu(512, 256, 3, 1)
        self.conv_up0 = convrelu(512, 128, 3, 1)


        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        layer0 = self.layer0(input)
        gate0 = self.gate_layer0(input)
        layer0 = self.sigmoid(gate0) * layer0
        
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer2 = self.aspp(layer2)

        x = self.upsample(layer2)                      # 256
        layer1 = self.layer1_1x1(layer1)                    # 64

        x = torch.cat([x, layer1], dim=1)                           # 256 + 64
        x = self.conv_up1(x)                                        # 256

        x = self.upsample(x)                                       # 256 * 100 * 200
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet18()
    model = model.to(device)

    summary(model, input_size=(3, 400, 400))