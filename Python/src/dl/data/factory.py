"""
数据加载工厂模块。

根据 ExperimentConfig 构建 train/val/test 的 DataLoader，
支持 MNIST、CIFAR10、CSV 回归/分类等数据集。
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from dl.config import ExperimentConfig
from dl.data.tabular import TabularDatasets, build_csv
from dl.data.vision import VisionDatasets, build_cifar10, build_mnist


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    in_features: int | None
    num_classes: int | None
    channels: int | None
    image_size: int | None


def build_dataloaders(cfg: ExperimentConfig) -> DataBundle:
    """根据配置构建训练/验证/测试 DataLoader 及元信息（channels、image_size、in_features 等）。"""
    dcfg = cfg.data
    task = cfg.task

    if dcfg.name.lower() == "mnist":
        ds: VisionDatasets = build_mnist(
            data_dir=dcfg.data_dir,
            image_size=dcfg.image_size,
            val_split=dcfg.val_split,
            seed=cfg.train.seed,
        )
        train_loader = DataLoader(
            ds.train,
            batch_size=dcfg.batch_size,
            shuffle=True,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            ds.val,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            ds.test,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            in_features=None,
            num_classes=ds.num_classes,
            channels=ds.channels,
            image_size=ds.image_size,
        )

    if dcfg.name.lower() == "cifar10":
        ds = build_cifar10(
            data_dir=dcfg.data_dir,
            image_size=dcfg.image_size,
            val_split=dcfg.val_split,
            seed=cfg.train.seed,
        )
        train_loader = DataLoader(
            ds.train,
            batch_size=dcfg.batch_size,
            shuffle=True,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            ds.val,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            ds.test,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            in_features=None,
            num_classes=ds.num_classes,
            channels=ds.channels,
            image_size=ds.image_size,
        )

    if dcfg.name.lower() in {"csv_regression", "csv_classification"}:
        if dcfg.csv_path is None or dcfg.target_col is None:
            raise ValueError("csv_path and target_col must be set for CSV datasets")
        tab_task = "regression" if dcfg.name.lower() == "csv_regression" else "classification"
        if task not in {"regression", "classification"}:
            raise ValueError(f"Unknown experiment task: {task}")
        if task == "regression" and tab_task != "regression":
            raise ValueError("cfg.task=regression but data.name=csv_classification")
        if task == "classification" and tab_task != "classification":
            raise ValueError("cfg.task=classification but data.name=csv_regression")

        ds: TabularDatasets = build_csv(
            csv_path=dcfg.csv_path,
            target_col=dcfg.target_col,
            feature_cols=dcfg.feature_cols,
            task=tab_task,
            val_split=dcfg.val_split,
            seed=cfg.train.seed,
        )
        train_loader = DataLoader(
            ds.train, batch_size=dcfg.batch_size, shuffle=True, num_workers=dcfg.num_workers
        )
        val_loader = DataLoader(
            ds.val, batch_size=dcfg.batch_size, shuffle=False, num_workers=dcfg.num_workers
        )
        test_loader = DataLoader(
            ds.test, batch_size=dcfg.batch_size, shuffle=False, num_workers=dcfg.num_workers
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            in_features=ds.info.in_features,
            num_classes=ds.info.num_classes,
            channels=None,
            image_size=None,
        )

    raise ValueError(f"Unknown dataset: {dcfg.name}")

