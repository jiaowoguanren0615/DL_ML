import torch
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, image_path, image_labels, transforms=None):
        self.image_path = image_path
        self.image_labels = image_labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        label = self.image_labels[item]
        if self.transforms:
            image = self.transforms(image)

        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels