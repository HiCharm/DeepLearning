"""
可视化模块。

绘制训练/验证的 loss、acc 等曲线，按后缀（_loss、_acc 等）分组。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history: dict[str, list[float]], out_path: str | Path, title: str = "") -> None:
    """
    绘制训练历史曲线。history 示例键：train_loss、val_loss、train_acc、val_acc。
    按后缀（_loss、_acc）分组，每个组一个子图。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(history.keys())
    if not keys:
        return

    # group by suffix (_loss/_acc) for nicer plots
    groups: dict[str, list[str]] = {}
    for k in keys:
        suffix = k.split("_")[-1] if "_" in k else k
        groups.setdefault(suffix, []).append(k)

    n = len(groups)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7, 4 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (suffix, gkeys) in zip(axes, groups.items()):
        for k in sorted(gkeys):
            ax.plot(history[k], label=k)
        ax.set_xlabel("epoch")
        ax.set_ylabel(suffix)
        ax.grid(True, alpha=0.3)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

