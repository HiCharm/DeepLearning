"""
表格数据模块。

支持从 CSV 加载回归/分类数据，自动划分训练/验证集。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


@dataclass(frozen=True)
class TabularInfo:
    in_features: int
    num_classes: int | None  # classification only


class CSVDataset(Dataset):
    """PyTorch 数据集：从 CSV 加载特征与标签，支持回归和分类任务。"""
    def __init__(
        self,
        csv_path: str | Path,
        target_col: str,
        feature_cols: list[str] | None,
        task: str,  # regression | classification
    ) -> None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

        df = pd.read_csv(p)
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in columns: {list(df.columns)}")

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols missing in CSV: {missing}")

        self.task = task
        x = df[feature_cols].astype("float32").to_numpy()

        if task == "regression":
            y = df[target_col].astype("float32").to_numpy()
            y = y.reshape(-1, 1)
            self.y = torch.from_numpy(y)
            self.num_classes = None
        elif task == "classification":
            # assume integer class labels in CSV
            y = df[target_col].astype("int64").to_numpy()
            self.y = torch.from_numpy(y)
            self.num_classes = int(self.y.max().item() + 1) if len(self.y) else 0
        else:
            raise ValueError(f"Unknown task: {task}")

        self.x = torch.from_numpy(x)
        self.in_features = self.x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass(frozen=True)
class TabularDatasets:
    """表格数据集的 train/val/test 划分及元信息。"""
    train: Dataset
    val: Dataset
    test: Dataset
    info: TabularInfo


def build_csv(
    csv_path: str | Path,
    target_col: str,
    feature_cols: list[str] | None,
    task: str,  # regression | classification
    val_split: float,
    seed: int,
) -> TabularDatasets:
    """从 CSV 构建 TabularDatasets，按 val_split 划分 train/val，test 复用 val。"""
    ds = CSVDataset(
        csv_path=csv_path, target_col=target_col, feature_cols=feature_cols, task=task
    )
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1)")
    n = len(ds)
    n_val = int(round(n * val_split))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train, val = random_split(ds, [n_train, n_val], generator=gen)
    # tabular demo: no dedicated test split -> reuse val as test
    test = val
    return TabularDatasets(
        train=train, val=val, test=test, info=TabularInfo(ds.in_features, ds.num_classes)
    )

