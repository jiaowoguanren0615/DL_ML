import torch
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_path, image_labels, transform=None):
        self.image_path = image_path
        self.image_labels = image_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        label = self.image_labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels