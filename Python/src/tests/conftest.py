"""
pytest 共享配置。

将 src 目录加入模块搜索路径，便于测试中导入 dl 包。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

