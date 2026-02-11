"""
训练脚本入口。

负责加载配置、构建数据加载器与模型、执行训练循环、评估、保存检查点与指标。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# 允许从 Python/ 目录执行 `python train.py`，将 src 加入模块搜索路径
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from dl.config import load_config, save_config  # noqa: E402
from dl.data.factory import build_dataloaders  # noqa: E402
from dl.io.paths import prepare_run_dir  # noqa: E402
from dl.models.factory import build_model  # noqa: E402
from dl.training.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from dl.training.trainer import Trainer  # noqa: E402
from dl.utils.device import resolve_device  # noqa: E402
from dl.utils.logging import setup_logging  # noqa: E402
from dl.utils.seed import set_seed  # noqa: E402
from dl.viz.plot import plot_history  # noqa: E402


def build_optimizer(cfg_name: str, params, lr: float, weight_decay: float, momentum: float):
    """根据配置名称构建优化器（Adam 或 SGD）。"""
    name = cfg_name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {cfg_name}")


def main() -> int:
    """主流程：加载配置、构建组件、训练、评估、保存结果。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/mnist_lenet.yaml", help="配置文件路径")
    args = ap.parse_args()

    cfg = load_config(THIS_DIR / args.config)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    paths = prepare_run_dir(cfg.output_dir, cfg.run_name)
    logger = setup_logging(paths.run_dir)
    logger.info(f"Run dir: {paths.run_dir}")
    logger.info(f"Device: {device}")

    save_config(cfg, paths.run_dir / "config.yaml")

    data = build_dataloaders(cfg)
    model_bundle = build_model(cfg, data)
    model = model_bundle.model

    optimizer = build_optimizer(
        cfg.optim.name,
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )

    if model_bundle.task == "classification":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    callbacks = [
        ModelCheckpoint(ckpt_dir=paths.ckpt_dir, monitor="val_loss", mode="min", save_last=True)
    ]
    if cfg.train.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                patience=cfg.train.early_stopping_patience, monitor="val_loss", mode="min"
            )
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        task=model_bundle.task,
        device=device,
        amp=cfg.train.amp,
        grad_clip_norm=cfg.train.grad_clip_norm,
        callbacks=callbacks,
        logger=logger,
        config_payload={"config_path": str(args.config)},
    )

    history = trainer.fit(
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        epochs=cfg.train.epochs,
        log_every=cfg.train.log_every,
        eval_every=cfg.train.eval_every,
    )

    test_result = trainer.evaluate(data.test_loader)
    logger.info(f"Test loss={test_result.loss:.4f} metrics={test_result.metrics}")

    plot_history(history, paths.fig_dir / "history.png", title=cfg.run_name)

    metrics_path = paths.run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"test_loss": test_result.loss, "test_metrics": test_result.metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved metrics to: {metrics_path}")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
