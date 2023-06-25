import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from .image_dataset import ImageDataset
from PIL import Image


def calculate_mean_and_std(data_dir: str):
    mean = np.zeros(1)
    std = np.zeros(1)
    count = 0
    for sub_dir in os.listdir(data_dir):
        sub_dir_full_path = os.path.join(data_dir, sub_dir)
        for image_path in os.listdir(sub_dir_full_path):
            image_full_path = os.path.join(sub_dir_full_path, image_path)
            image = np.array(Image.open(image_full_path))
            mean += np.mean(image, axis=(0, 1))
            std += np.std(image, axis=(0, 1))
            count += 1
    mean /= count * 255.0
    std /= count * 255.0
    return mean, std


def build_transforms(data_dir: str, is_augmented=False) -> transforms:
    means, stds = calculate_mean_and_std(data_dir)
    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )
    if is_augmented:
        data_transforms = transforms.Compose(
            [
                data_transforms,
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.2),
            ]
        )
    return data_transforms


def build_dataloader(
    data_dir: str, batch_size: int, transforms: transforms
) -> DataLoader:
    data_transforms = transforms
    image_dataset = ImageDataset(data_dir=data_dir, transforms=data_transforms)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return image_dataloader
