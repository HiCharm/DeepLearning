"""
线性模型模块。

包含 LinearRegression（回归）和 LogisticRegression（分类）。
"""
from __future__ import annotations

import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """线性回归：单层全连接，输出维度 1。"""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.linear(x)


class LogisticRegressionModel(nn.Module):
    """逻辑回归（多分类）：单层全连接，输出维度 num_classes。"""
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.linear(x)

