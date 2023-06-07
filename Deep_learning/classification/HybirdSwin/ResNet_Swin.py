from ResNet50model import make_resnet50
import torch.nn as nn
from torchinfo import summary
import warnings
from swin import swin_tiny_patch4_window7_224


warnings.filterwarnings('ignore')

class Res50Base(nn.Sequential):
    def __init__(self):
        super(Res50Base, self).__init__(
            make_resnet50()
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=1024, out_channels=96, kernel_size=1, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )


class Swin_Transformer(nn.Sequential):
    def __init__(self,  num_classes):
        super(Swin_Transformer, self).__init__(
            swin_tiny_patch4_window7_224(num_classes=num_classes)
        )


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = Res50Base()
        self.layer2 = ConvBNReLU()
        self.layer3 = Swin_Transformer(num_classes=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.layer3(x)
        return y

# net = NeuralNetwork(num_classes=1000)
# summary(net, input_size=(1, 3, 224, 224))