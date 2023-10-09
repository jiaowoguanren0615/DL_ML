import torch
import torch.nn as nn
from config import cfgs

def make_features(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1) # 3 64
            layers += [conv2d, nn.BatchNorm2d(num_features=v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    def __init__(self, features, num_classes, init_weights=False) -> None:
        super().__init__()
        self.features = features
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()
    
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) #[B, C, H, W]
        y = self.classifier(x)
        return y
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    
def make_model(model_name='vgg16', num_classes=1000):
    cfg = cfgs[model_name]
    net = VGGNet(make_features(cfg), num_classes=num_classes, init_weights=True)
    return net

# net = make_model()
# summary(net, input_size=(1, 3, 224, 224))