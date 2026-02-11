"""
训练器冒烟测试。

验证 Trainer 在分类/回归任务下能正常 fit 并返回预期 history。
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from dl.training.trainer import Trainer


def test_trainer_one_epoch_classification_smoke() -> None:
    """分类任务：2 个 epoch，history 含 train_acc/val_acc。"""
    torch.manual_seed(0)
    x = torch.randn(64, 10)
    y = torch.randint(0, 3, (64,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    t = Trainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        task="classification",
        device=torch.device("cpu"),
        amp=False,
    )
    hist = t.fit(loader, loader, epochs=2, log_every=1000, eval_every=1)
    assert len(hist["train_loss"]) == 2
    assert len(hist["val_loss"]) == 2
    assert "train_acc" in hist


def test_trainer_one_epoch_regression_smoke() -> None:
    """回归任务：2 个 epoch，history 含 train_mse/val_mse。"""
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x[:, :1] * 2.0 + 1.0).clone()
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = torch.nn.Linear(4, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-1)
    loss_fn = torch.nn.MSELoss()

    t = Trainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        task="regression",
        device=torch.device("cpu"),
        amp=False,
    )
    hist = t.fit(loader, loader, epochs=2, log_every=1000, eval_every=1)
    assert len(hist["train_loss"]) == 2
    assert len(hist["val_loss"]) == 2
    assert "train_mse" in hist

