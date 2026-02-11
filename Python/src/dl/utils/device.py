"""
设备选择模块。

根据配置字符串（auto/cuda/cpu）解析实际使用的 torch.device。
"""
from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """将 'auto'/'cuda'/'cpu' 转为 torch.device，auto 表示有 GPU 则用 cuda。"""
    d = device.lower().strip()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d in {"cuda", "cpu"}:
        if d == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(d)
    raise ValueError(f"Unknown device: {device}")

