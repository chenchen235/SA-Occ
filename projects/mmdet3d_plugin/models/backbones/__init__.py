from mmdet.models.backbones import ResNet
from .resnet import CustomResNet
from .swin import SwinTransformer
from .resunet18 import ResNetUNet18
from .resunet50 import ResNetUNet50

__all__ = ['ResNet', 'CustomResNet', 'SwinTransformer', 'ResNetUNet18', 'ResNetUNet50']
