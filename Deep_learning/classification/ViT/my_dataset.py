import torch
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_path, image_labels, transform=None):
        self.image_path = image_path
        self.image_class = image_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        # 转成RGB格式
        img = Image.open(self.image_path[item]).convert('RGB')
        label = self.image_class[item]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # DataLoader 方法会用到 如果不设置 则使用官方默认的
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels