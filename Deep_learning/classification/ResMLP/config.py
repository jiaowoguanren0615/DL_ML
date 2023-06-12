import torch
from torchvision import transforms

root = r'D:/flower_data'
batch_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 5
lr = 0.001
weight_decay = 0.001

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


num_gpus = torch.cuda.device_count()
def try_gpu(i):
    assert torch.cuda.is_available()
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')

    return torch.device('cpu')


devices = [try_gpu(i) for i in range(num_gpus)]
