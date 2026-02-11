"""
兼容入口：保留原 dataloader 导入路径。

实际实现在 `dl.data.factory.build_dataloaders`。
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

from dl.config import ExperimentConfig  # noqa: E402
from dl.data.factory import DataBundle, build_dataloaders  # noqa: E402

__all__ = ["DataBundle", "build_dataloaders", "ExperimentConfig"]

