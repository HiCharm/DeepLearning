"""
模型工厂模块。

根据 ExperimentConfig 和 DataBundle 构建对应的 PyTorch 模型，
支持 lenet、resnet18、mlp、linear_regression、logistic_regression 等。
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from dl.config import ExperimentConfig
from dl.data.factory import DataBundle
from dl.models.cnn import LeNet5, SimpleCNN
from dl.models.linear import LinearRegressionModel, LogisticRegressionModel
from dl.models.mlp import MLP
from dl.models.resnet import resnet18


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    task: str  # classification | regression


def build_model(cfg: ExperimentConfig, data: DataBundle) -> ModelBundle:
    """根据配置与数据元信息构建模型，返回 ModelBundle（model + task 类型）。"""
    mcfg = cfg.model
    name = mcfg.name.lower()
    task = cfg.task

    if name == "lenet":
        if data.channels is None:
            raise ValueError("LeNet requires vision dataset (channels not found).")
        num_classes = mcfg.num_classes if data.num_classes is None else data.num_classes
        model = LeNet5(in_channels=data.channels, num_classes=num_classes)
        return ModelBundle(model=model, task="classification")

    if name == "simplecnn":
        if data.channels is None:
            raise ValueError("SimpleCNN requires vision dataset.")
        num_classes = mcfg.num_classes if data.num_classes is None else data.num_classes
        model = SimpleCNN(in_channels=data.channels, num_classes=num_classes)
        return ModelBundle(model=model, task="classification")

    if name == "resnet18":
        num_classes = mcfg.num_classes if data.num_classes is None else data.num_classes
        model = resnet18(num_classes=num_classes)
        return ModelBundle(model=model, task="classification")

    if name == "linear_regression":
        in_features = mcfg.in_features or data.in_features
        if in_features is None:
            raise ValueError("linear_regression requires in_features (from config or data).")
        model = LinearRegressionModel(in_features=in_features)
        return ModelBundle(model=model, task="regression")

    if name == "logistic_regression":
        in_features = mcfg.in_features or data.in_features
        if in_features is None:
            raise ValueError("logistic_regression requires in_features (from config or data).")
        num_classes = mcfg.num_classes if data.num_classes is None else data.num_classes
        model = LogisticRegressionModel(in_features=in_features, num_classes=num_classes)
        return ModelBundle(model=model, task="classification")

    if name == "mlp":
        in_features = mcfg.in_features or data.in_features
        if in_features is None:
            # vision mlp: flatten image
            if data.channels is None or data.image_size is None:
                raise ValueError("MLP requires tabular in_features or vision metadata.")
            in_features = data.channels * data.image_size * data.image_size

        if task == "classification":
            num_classes = mcfg.num_classes if data.num_classes is None else data.num_classes
            model = MLP(
                in_features=in_features,
                out_features=num_classes,
                hidden_sizes=mcfg.hidden_sizes,
                dropout=mcfg.dropout,
            )
            return ModelBundle(model=model, task="classification")

        if task == "regression":
            model = MLP(
                in_features=in_features,
                out_features=1,
                hidden_sizes=mcfg.hidden_sizes,
                dropout=mcfg.dropout,
            )
            return ModelBundle(model=model, task="regression")

        raise ValueError(f"Unknown task: {task}")

    raise ValueError(f"Unknown model: {mcfg.name}")

