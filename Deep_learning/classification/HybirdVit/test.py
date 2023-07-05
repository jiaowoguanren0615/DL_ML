# 以vit_large_patch16_224为例
from ResNet50model import make_resnet50
from torchinfo import summary
import torch
import torch.nn as nn
from ViTmodel import vit_large_patch16_224

class Res50Base(nn.Sequential):
    def __init__(self):
        super(Res50Base, self).__init__(
            make_resnet50()
        )



class ViT(nn.Sequential):
    def __init__(self,  num_classes):
        super(ViT, self).__init__(
            vit_large_patch16_224(num_classes=num_classes)
        )

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = Res50Base()
        self.layer3 = ViT(num_classes=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        y = self.layer3(x)
        return y


net = NeuralNetwork(num_classes=5)
summary(net, input_size=(1, 3, 224, 224))