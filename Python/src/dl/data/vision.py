"""
图像数据集模块。

支持 MNIST、CIFAR10，包含 Resize、Normalize 等 transform，并划分 train/val/test。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class VisionDatasets:
    train: Dataset
    val: Dataset
    test: Dataset
    num_classes: int
    channels: int
    image_size: int


def _split_train_val(ds: Dataset, val_split: float, seed: int) -> tuple[Dataset, Dataset]:
    """将完整训练集按 val_split 比例划分为 train/val。"""
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1)")
    n = len(ds)
    n_val = int(round(n * val_split))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=gen)


def build_mnist(data_dir: str, image_size: int, val_split: float, seed: int) -> VisionDatasets:
    """构建 MNIST 数据集（28×28 灰度图，10 类）。"""
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    train, val = _split_train_val(train_full, val_split=val_split, seed=seed)
    return VisionDatasets(
        train=train, val=val, test=test, num_classes=10, channels=1, image_size=image_size
    )


def build_cifar10(data_dir: str, image_size: int, val_split: float, seed: int) -> VisionDatasets:
    """构建 CIFAR10 数据集（32×32 彩色图，10 类，训练时带随机水平翻转）。"""
    tfm_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm_test)
    train, val = _split_train_val(train_full, val_split=val_split, seed=seed)
    return VisionDatasets(
        train=train, val=val, test=test, num_classes=10, channels=3, image_size=image_size
    )

