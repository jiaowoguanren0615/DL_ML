import torch
from PIL import Image
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        label = self.labels[item]

        if self.transforms:
            image = self.transforms(image)
        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def build_transform(is_train, input_size=224, color_jitter=0.4,
                    aa='rand-m9-mstd0.5-inc1', reprob=0.25, finetune=False):
    resize_im = input_size > 32

    train_interpolation = ['random', 'bilinear', 'bicubic']
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation=train_interpolation[2],
            re_prob=reprob,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if finetune:
        t.append(
            transforms.Resize((input_size, input_size),
                              interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)