"""
日志模块。

配置控制台与文件日志（run.log），输出到指定目录。
"""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str | Path, name: str = "dl") -> logging.Logger:
    """创建 logger，同时输出到控制台和 log_dir/run.log。"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

