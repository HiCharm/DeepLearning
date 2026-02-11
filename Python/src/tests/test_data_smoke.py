"""
数据模块冒烟测试。

验证 CSV 回归数据加载器能正确构建并产出符合预期的 batch。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dl.config import DataConfig, ExperimentConfig
from dl.data.factory import build_dataloaders


def test_csv_regression_dataloader_smoke(tmp_path: Path) -> None:
    """测试 CSV 回归数据集加载：特征维度 2，标签维度 1。"""
    csv_path = tmp_path / "reg.csv"
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 1.0, 1.0, 1.0], "y": [1.0, 3.0, 5.0, 7.0]})
    df.to_csv(csv_path, index=False)

    cfg = ExperimentConfig(
        task="regression",
        data=DataConfig(
            name="csv_regression",
            csv_path=str(csv_path),
            target_col="y",
            batch_size=2,
            num_workers=0,
            val_split=0.5,
        ),
    )
    bundle = build_dataloaders(cfg)
    x, y = next(iter(bundle.train_loader))
    assert x.shape[-1] == 2
    assert y.shape[-1] == 1

