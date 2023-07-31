import torch
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_path, image_labels, transforms=None, output_label=True):

        self.image_path = image_path
        self.image_labels = image_labels
        self.transforms = transforms
        self.output_label = output_label

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        if self.output_label:
            label = self.image_labels[item]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_path)

    @staticmethod
    def collate_fn(batch):
        image, label = tuple(zip(*batch))
        image = torch.stack(image, dim=0)
        label = torch.as_tensor(label)
        return image, label