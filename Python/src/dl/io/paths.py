"""
运行路径管理模块。

为每次实验创建带时间戳的运行目录，并管理 checkpoints、figures 等子目录。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_dir: Path
    fig_dir: Path


def prepare_run_dir(output_dir: str | Path, run_name: str) -> RunPaths:
    """创建 {run_name}-{YYYYMMDD-HHMMSS} 运行目录及 checkpoints、figures 子目录。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{run_name}-{ts}"
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, ckpt_dir=ckpt_dir, fig_dir=fig_dir)

