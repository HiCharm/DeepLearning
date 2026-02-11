"""
推理/预测可视化脚本。

加载训练好的检查点，对测试集样本进行预测并可视化（预测值 vs 真实标签）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# 模块路径设置，便于从 Python/ 目录执行
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from dl.config import load_config  # noqa: E402
from dl.data.factory import build_dataloaders  # noqa: E402
from dl.io.checkpoints import load_checkpoint  # noqa: E402
from dl.io.paths import prepare_run_dir  # noqa: E402
from dl.models.factory import build_model  # noqa: E402
from dl.utils.device import resolve_device  # noqa: E402


def _denormalize(name: str, x: torch.Tensor) -> torch.Tensor:
    """
    根据数据集名称将归一化后的图像反归一化，便于可视化。
    x: [C,H,W] 格式的张量
    """
    n = name.lower()
    if n == "mnist":
        mean = torch.tensor([0.1307], device=x.device)[:, None, None]
        std = torch.tensor([0.3081], device=x.device)[:, None, None]
        return x * std + mean
    if n == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device)[:, None, None]
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device)[:, None, None]
        return x * std + mean
    return x


@torch.no_grad()
def main() -> int:
    """主流程：加载配置与检查点，推理并保存可视化图。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/mnist_lenet.yaml")
    ap.add_argument("--checkpoint", type=str, default=None, help="path to best.pt/last.pt")
    ap.add_argument("--n", type=int, default=16, help="number of samples to visualize")
    args = ap.parse_args()

    cfg = load_config(THIS_DIR / args.config)
    device = resolve_device(cfg.train.device)

    data = build_dataloaders(cfg)
    model_bundle = build_model(cfg, data)
    model = model_bundle.model.to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # fallback: create a new run dir name and try to locate nothing
        raise ValueError("Please pass --checkpoint (e.g. outputs/<run>/checkpoints/best.pt)")

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x, y = next(iter(data.test_loader))
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    if model_bundle.task != "classification":
        raise ValueError("infer.py visualization currently supports classification tasks only.")
    preds = torch.argmax(logits, dim=1)

    n = min(args.n, x.shape[0])
    cols = int((n**0.5) + 0.999)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        img = _denormalize(cfg.data.name, x[i]).detach().cpu().clamp(0, 1)
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"p={int(preds[i])} / y={int(y[i])}", fontsize=9)

    # save alongside checkpoint if possible; else into a new run dir
    ckpt_p = Path(ckpt_path)
    out_dir = ckpt_p.parent.parent if ckpt_p.name in {"best.pt", "last.pt"} else prepare_run_dir(
        cfg.output_dir, cfg.run_name
    ).run_dir
    out_path = Path(out_dir) / "infer_predictions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
