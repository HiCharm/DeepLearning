"""
检查点模块。

保存/加载模型权重、优化器状态、epoch、metrics 等，便于恢复训练或推理。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Checkpoint:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    epoch: int
    metrics: dict[str, float]
    config: dict[str, Any] | None = None


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """将 payload（含 model_state、optimizer_state 等）保存到指定路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """加载检查点到指定设备。"""
    return torch.load(Path(path), map_location=map_location)

