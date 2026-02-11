"""
模型前向传播形状测试。

验证各模型在给定输入 shape 下输出 shape 正确。
"""
from __future__ import annotations

import torch

from dl.models.cnn import LeNet5, SimpleCNN
from dl.models.linear import LinearRegressionModel, LogisticRegressionModel
from dl.models.mlp import MLP
from dl.models.resnet import resnet18


def test_lenet_forward_shape() -> None:
    """LeNet5：输入 [2,1,28,28] -> 输出 [2,10]。"""
    m = LeNet5(in_channels=1, num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    assert y.shape == (2, 10)


def test_simplecnn_forward_shape() -> None:
    """SimpleCNN：输入 [2,3,32,32] -> 输出 [2,10]。"""
    m = SimpleCNN(in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    assert y.shape == (2, 10)


def test_resnet18_forward_shape() -> None:
    """ResNet18：输入 [2,3,32,32] -> 输出 [2,10]。"""
    m = resnet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    assert y.shape == (2, 10)


def test_mlp_forward_shape() -> None:
    """MLP：输入 [4,20] -> 输出 [4,3]。"""
    m = MLP(in_features=20, out_features=3, hidden_sizes=[16, 8], dropout=0.1)
    x = torch.randn(4, 20)
    y = m(x)
    assert y.shape == (4, 3)


def test_linear_and_logistic_shapes() -> None:
    """线性回归输出 [N,1]，逻辑回归输出 [N,num_classes]。"""
    lin = LinearRegressionModel(in_features=5)
    log = LogisticRegressionModel(in_features=5, num_classes=7)
    x = torch.randn(3, 5)
    assert lin(x).shape == (3, 1)
    assert log(x).shape == (3, 7)

