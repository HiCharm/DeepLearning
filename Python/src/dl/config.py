"""
配置管理模块。

定义实验配置的数据类（DataConfig、ModelConfig、OptimConfig、TrainConfig、ExperimentConfig），
支持从 YAML 加载配置并合并默认值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """数据相关配置。"""
    name: str = "mnist"  # 数据集名称：mnist | cifar10 | csv_regression | csv_classification
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1

    # vision
    image_size: int = 28

    # csv
    csv_path: str | None = None
    target_col: str | None = None
    feature_cols: list[str] | None = None


@dataclass(frozen=True)
class ModelConfig:
    """模型相关配置。"""
    name: str = "lenet"  # 模型名称：lenet | mlp | logistic_regression | linear_regression | resnet18 等
    num_classes: int = 10
    in_features: int | None = None  # for tabular models
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.0


@dataclass(frozen=True)
class OptimConfig:
    """优化器相关配置。"""
    name: str = "adam"  # 优化器：adam | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


@dataclass(frozen=True)
class TrainConfig:
    """训练相关配置。"""
    epochs: int = 5
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    amp: bool = False
    grad_clip_norm: float | None = None

    log_every: int = 50
    eval_every: int = 1  # epochs

    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """完整实验配置，聚合 data/model/optim/train 等子配置。"""
    project: str = "DeepLearning-PyTorch"
    run_name: str = "run"
    output_dir: str = "./outputs"
    task: str = "classification"  # classification | regression

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _as_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    raise TypeError(f"Expected dict, got {type(x)}")


def _merge_dataclass(dc_type: type[Any], base: Any, updates: dict[str, Any]) -> Any:
    merged = {**base.__dict__, **updates}
    return dc_type(**merged)


def load_config(path: str | Path) -> ExperimentConfig:
    """从 YAML 文件加载配置，与默认值合并后返回 ExperimentConfig。"""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = _as_dict(raw)

    base = ExperimentConfig()
    data = _merge_dataclass(DataConfig, base.data, _as_dict(raw.get("data", {})))
    model = _merge_dataclass(ModelConfig, base.model, _as_dict(raw.get("model", {})))
    optim = _merge_dataclass(OptimConfig, base.optim, _as_dict(raw.get("optim", {})))
    train = _merge_dataclass(TrainConfig, base.train, _as_dict(raw.get("train", {})))

    top_updates = {k: v for k, v in raw.items() if k not in {"data", "model", "optim", "train"}}
    exp = _merge_dataclass(ExperimentConfig, base, top_updates)
    return ExperimentConfig(
        project=exp.project,
        run_name=exp.run_name,
        output_dir=exp.output_dir,
        task=exp.task,
        data=data,
        model=model,
        optim=optim,
        train=train,
    )


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """将实验配置保存为 YAML 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": cfg.project,
        "run_name": cfg.run_name,
        "output_dir": cfg.output_dir,
        "task": cfg.task,
        "data": cfg.data.__dict__,
        "model": cfg.model.__dict__,
        "optim": cfg.optim.__dict__,
        "train": cfg.train.__dict__,
    }
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

