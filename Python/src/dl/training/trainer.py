"""
训练器模块。

封装训练循环、验证、评估逻辑，支持 AMP、梯度裁剪、EarlyStopping、ModelCheckpoint 等。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl.training.callbacks import Callback, EarlyStopping
from dl.training.metrics import accuracy, mae, mse


@dataclass(frozen=True)
class EvalResult:
    loss: float
    metrics: dict[str, float]


class Trainer:
    """训练器：负责 fit（训练+验证）和 evaluate（评估），支持回调与 AMP。"""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        task: str,  # classification | regression
        device: torch.device,
        amp: bool = False,
        grad_clip_norm: float | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        config_payload: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task = task
        self.device = device
        self.amp = amp
        self.grad_clip_norm = grad_clip_norm
        self.callbacks = callbacks or []
        self.logger = logger
        self.config_payload = config_payload

        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        log_every: int = 50,
        eval_every: int = 1,
    ) -> dict[str, list[float]]:
        self.model.to(self.device)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        if self.task == "classification":
            history["train_acc"] = []
            history["val_acc"] = []
        else:
            history["train_mse"] = []
            history["val_mse"] = []
            history["train_mae"] = []
            history["val_mae"] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            n_seen = 0

            if self.task == "classification":
                correct = 0
                total = 0
            else:
                # accumulate regression metrics with sums over batch
                mse_sum = 0.0
                mae_sum = 0.0

            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
            for step, (x, y) in enumerate(pbar, start=1):
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.amp):
                    out = self.model(x)
                    loss = self.loss_fn(out, y)

                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs = x.shape[0]
                train_loss_sum += float(loss.item()) * bs
                n_seen += bs

                if self.task == "classification":
                    preds = torch.argmax(out.detach(), dim=1)
                    correct += int((preds == y).sum().item())
                    total += int(y.numel())
                    if step % log_every == 0:
                        pbar.set_postfix(loss=float(loss.item()), acc=correct / max(total, 1))
                else:
                    preds = out.detach()
                    mse_sum += float(torch.mean((preds - y) ** 2).item()) * bs
                    mae_sum += float(torch.mean(torch.abs(preds - y)).item()) * bs
                    if step % log_every == 0:
                        pbar.set_postfix(loss=float(loss.item()))

            train_loss = train_loss_sum / max(n_seen, 1)
            history["train_loss"].append(train_loss)

            logs: dict[str, float] = {"train_loss": train_loss}
            if self.task == "classification":
                train_acc = correct / max(total, 1)
                history["train_acc"].append(train_acc)
                logs["train_acc"] = float(train_acc)
            else:
                train_mse = mse_sum / max(n_seen, 1)
                train_mae = mae_sum / max(n_seen, 1)
                history["train_mse"].append(train_mse)
                history["train_mae"].append(train_mae)
                logs["train_mse"] = float(train_mse)
                logs["train_mae"] = float(train_mae)

            if eval_every > 0 and (epoch % eval_every == 0):
                val = self.evaluate(val_loader)
                history["val_loss"].append(val.loss)
                logs["val_loss"] = float(val.loss)
                for k, v in val.metrics.items():
                    history.setdefault(f"val_{k}", []).append(v)
                    logs[f"val_{k}"] = float(v)
            else:
                # keep arrays aligned
                history["val_loss"].append(float("nan"))

            self._log(
                f"[epoch {epoch}/{epochs}] "
                + " ".join([f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float)])
            )

            state = {
                "model": self.model,
                "optimizer": self.optimizer,
                "config": self.config_payload,
            }
            for cb in self.callbacks:
                cb.on_epoch_end(epoch=epoch, logs=logs, state=state)

            for cb in self.callbacks:
                if isinstance(cb, EarlyStopping) and cb.should_stop:
                    self._log(f"EarlyStopping triggered at epoch {epoch}.")
                    return history

        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvalResult:
        self.model.eval()
        loss_sum = 0.0
        n_seen = 0

        if self.task == "classification":
            acc_sum = 0.0
        else:
            mse_sum = 0.0
            mae_sum = 0.0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x)
            loss = self.loss_fn(out, y)

            bs = x.shape[0]
            loss_sum += float(loss.item()) * bs
            n_seen += bs

            if self.task == "classification":
                acc_sum += accuracy(out, y) * bs
            else:
                mse_sum += mse(out, y) * bs
                mae_sum += mae(out, y) * bs

        loss_avg = loss_sum / max(n_seen, 1)
        if self.task == "classification":
            return EvalResult(loss=loss_avg, metrics={"acc": acc_sum / max(n_seen, 1)})
        return EvalResult(
            loss=loss_avg,
            metrics={"mse": mse_sum / max(n_seen, 1), "mae": mae_sum / max(n_seen, 1)},
        )

