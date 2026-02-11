"""
训练回调模块。

EarlyStopping：监控指标无提升时提前停止；ModelCheckpoint：保存最佳与最新检查点。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dl.io.checkpoints import save_checkpoint


class Callback:
    """回调基类，子类可在 on_epoch_end 中实现自定义逻辑。"""
    def on_epoch_end(self, epoch: int, logs: dict[str, float], state: dict[str, Any]) -> None:
        pass


@dataclass
class EarlyStopping(Callback):
    """监控指定指标，连续 patience 个 epoch 无提升则停止训练。"""
    patience: int
    monitor: str = "val_loss"
    mode: str = "min"  # min | max
    min_delta: float = 0.0

    best: float | None = None
    bad_epochs: int = 0
    should_stop: bool = False

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)

    def on_epoch_end(self, epoch: int, logs: dict[str, float], state: dict[str, Any]) -> None:
        if self.monitor not in logs:
            return
        value = float(logs[self.monitor])
        if self._is_improvement(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True


@dataclass
class ModelCheckpoint(Callback):
    """每个 epoch 保存 last.pt，指标最优时保存 best.pt。"""
    ckpt_dir: str | Path
    monitor: str = "val_loss"
    mode: str = "min"
    save_last: bool = True

    best: float | None = None
    best_path: Path | None = None

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def on_epoch_end(self, epoch: int, logs: dict[str, float], state: dict[str, Any]) -> None:
        ckpt_dir = Path(self.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.save_last:
            save_checkpoint(
                ckpt_dir / "last.pt",
                {
                    "epoch": epoch,
                    "metrics": logs,
                    "model_state": state["model"].state_dict(),
                    "optimizer_state": state["optimizer"].state_dict() if state.get("optimizer") else None,
                    "config": state.get("config"),
                },
            )

        if self.monitor not in logs:
            return
        value = float(logs[self.monitor])
        if self._is_improvement(value):
            self.best = value
            self.best_path = ckpt_dir / "best.pt"
            save_checkpoint(
                self.best_path,
                {
                    "epoch": epoch,
                    "metrics": logs,
                    "model_state": state["model"].state_dict(),
                    "optimizer_state": state["optimizer"].state_dict() if state.get("optimizer") else None,
                    "config": state.get("config"),
                },
            )

