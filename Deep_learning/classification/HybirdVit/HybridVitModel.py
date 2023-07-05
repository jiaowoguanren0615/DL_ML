from ResNet50model import make_resnet50
from torchinfo import summary
import torch
import torch.nn as nn
from ViTmodel import vit_base_patch16_224_in21k


class Res50Base(nn.Sequential):
    def __init__(self):
        super(Res50Base, self).__init__(
            make_resnet50()
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1, stride=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )


class ViT(nn.Sequential):
    def __init__(self,  num_classes):
        super(ViT, self).__init__(
            vit_base_patch16_224_in21k(num_classes=num_classes)
        )


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = Res50Base()
        self.layer2 = ConvBNReLU()
        self.layer3 = ViT(num_classes=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.layer3(x)
        return y

