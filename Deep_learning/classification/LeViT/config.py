import torch
from torchvision import transforms

root = r'/mnt/d/flower_data'
batch_size = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 5
lr = 0.0003
best_val_accuracy = 0
weight_decay = 0.00001

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}