import os
import torch
from torchvision import transforms

root = '/usr/local/Huangshuqi/ImageData/flower_data'
batch_size = 32
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 2
best_val_accuracy = 0
lr = 0.003
weight_decay = 5e-4

save_path = './VGGModel.pth'
resume = True if os.path.exists(save_path) else False

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

    'valid': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
