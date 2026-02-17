# DeepLearning 深度学习实验框架

@HiCharm

本项目是用于学习深度学习的 PyTorch 实验框架，包含工具库、数据集和模型算法，支持分类、回归任务，支持 MNIST、CIFAR10、CSV 表格数据，以及 RFID（按坐标文件名分类）等。

---

## 一、项目结构

```
DeepLearning/
├── data/                           # 数据目录
│   └── MNIST/
│       └── raw/                    # MNIST 原始数据（自动下载）
│           ├── train-images-idx3-ubyte
│           ├── train-labels-idx1-ubyte
│           ├── t10k-images-idx3-ubyte
│           └── t10k-labels-idx1-ubyte
│
├── outputs/                        # 实验输出根目录
│   └── {run_name}-{YYYYMMDD-HHMMSS}/   # 每次运行的带时间戳目录
│       ├── checkpoints/
│       │   ├── best.pt             # 验证集最优模型
│       │   └── last.pt             # 最新 epoch 模型
│       ├── figures/
│       │   └── history.png         # 训练曲线图
│       ├── config.yaml             # 本次运行使用的配置
│       ├── run.log                 # 运行日志
│       ├── metrics.json            # 测试集指标
│       └── infer_predictions.png   # 推理可视化（可选）
│
├── Python/                         # Python 项目根目录
│   └── src/
│       ├── train.py                # 训练入口脚本
│       ├── infer.py                # 推理/可视化脚本
│       ├── configs/                # 配置文件
│       │   ├── mnist_lenet.yaml    # MNIST + LeNet 示例
│       │   ├── cifar10_resnet.yaml # CIFAR10 + ResNet18 示例
│       │   └── csv_regression_linear.yaml  # CSV 回归示例
│       │
│       ├── dl/                     # 核心深度学习库
│       │   ├── __init__.py
│       │   ├── config.py           # 配置加载与保存
│       │   ├── data/               # 数据模块
│       │   │   ├── factory.py      # 数据加载工厂
│       │   │   ├── vision.py       # MNIST、CIFAR10
│       │   │   └── tabular.py      # CSV 回归/分类
│       │   ├── io/                 # 输入输出
│       │   │   ├── paths.py        # 运行目录管理
│       │   │   └── checkpoints.py  # 检查点保存/加载
│       │   ├── models/             # 模型
│       │   │   ├── factory.py      # 模型工厂
│       │   │   ├── cnn.py          # LeNet5、SimpleCNN
│       │   │   ├── resnet.py       # ResNet18
│       │   │   ├── mlp.py          # 多层感知机
│       │   │   └── linear.py       # 线性/逻辑回归
│       │   ├── training/           # 训练
│       │   │   ├── trainer.py      # 训练器
│       │   │   ├── metrics.py      # 准确率、MSE、MAE
│       │   │   └── callbacks.py    # EarlyStopping、ModelCheckpoint
│       │   ├── utils/              # 工具
│       │   │   ├── device.py       # GPU/CPU 选择
│       │   │   ├── logging.py      # 日志配置
│       │   │   └── seed.py         # 随机种子
│       │   └── viz/                # 可视化
│       │       └── plot.py         # 训练曲线绘制
│       │
│       ├── tests/                  # 单元测试
│       │   ├── conftest.py
│       │   ├── test_data_smoke.py
│       │   ├── test_models.py
│       │   └── test_trainer_smoke.py
│       │
│       ├── requirements.txt
│       └── pyproject.toml
│
└── README.md                       # 本文件
```

---

## 二、路径与数据流说明

### 2.1 数据路径

| 路径 | 说明 |
|------|------|
| `./data` | 默认数据根目录，在配置中由 `data.data_dir` 指定 |
| `./data/MNIST/raw/` | MNIST 自动下载存放位置 |
| `./data/`（CIFAR10） | CIFAR10 自动下载存放位置 |
| `csv_path` | CSV 数据集路径，由配置中 `data.csv_path` 指定 |

### 2.2 输出路径

| 路径 | 说明 |
|------|------|
| `output_dir` | 输出根目录，默认 `./outputs` |
| `{run_name}-{YYYYMMDD-HHMMSS}` | 每次运行的唯一目录 |
| `checkpoints/best.pt` | 验证集指标最优时保存的模型 |
| `checkpoints/last.pt` | 每个 epoch 更新的最新模型 |
| `figures/history.png` | 训练 loss/acc 曲线 |
| `run.log` | 运行日志 |
| `config.yaml` | 本次使用的完整配置 |
| `metrics.json` | 测试集 loss 与 metrics |

### 2.3 配置文件路径

配置文件位于 `Python/src/configs/`，通过 `--config` 传入，例如：

```bash
python train.py --config configs/mnist_lenet.yaml
```

---

## 三、完整流程说明

### 3.1 训练流程（train.py）

```
1. 解析命令行参数（--config）
       ↓
2. 加载 YAML 配置 (load_config)
       ↓
3. 设置随机种子 (set_seed)
       ↓
4. 解析设备 (resolve_device: auto/cuda/cpu)
       ↓
5. 创建运行目录 (prepare_run_dir)
   outputs/{run_name}-{YYYYMMDD-HHMMSS}/
       ↓
6. 初始化日志 (setup_logging)
       ↓
7. 保存配置到 run_dir/config.yaml
       ↓
8. 构建数据加载器 (build_dataloaders)
   - MNIST / CIFAR10 / csv_regression / csv_classification / rfid
       ↓
9. 构建模型 (build_model)
   - lenet / resnet18 / mlp / linear_regression / logistic_regression
       ↓
10. 构建优化器 (build_optimizer: Adam / SGD)
       ↓
11. 选择损失函数
    - 分类: CrossEntropyLoss
    - 回归: MSELoss
       ↓
12. 配置回调 (ModelCheckpoint, EarlyStopping)
       ↓
13. 创建 Trainer，执行 fit()
    - 遍历 epoch
    - 每个 epoch: 训练 → 验证 → 回调（保存 checkpoint、早停）
       ↓
14. 测试集评估 (evaluate)
       ↓
15. 绘制 history 曲线 → figures/history.png
       ↓
16. 保存 metrics.json
```

### 3.2 推理流程（infer.py）

```
1. 解析命令行（--config, --checkpoint, --n）
       ↓
2. 加载配置
       ↓
3. 构建数据加载器与模型
       ↓
4. 加载检查点 (load_checkpoint)
   - 需指定 --checkpoint，如 outputs/mnist-lenet-xxx/checkpoints/best.pt
       ↓
5. 从 test_loader 取一批样本
       ↓
6. 模型推理，得到预测
       ↓
7. 反归一化图像，绘制 p=预测/y=真实 对比图
       ↓
8. 保存 infer_predictions.png 到 run_dir 或 checkpoint 父目录
```

### 3.3 模块调用关系

```
train.py / infer.py
    ├── dl.config (load_config, save_config)
    ├── dl.data.factory (build_dataloaders)
    │       ├── dl.data.vision (build_mnist, build_cifar10)
    │       └── dl.data.tabular (build_csv)
    ├── dl.models.factory (build_model)
    │       ├── dl.models.cnn (LeNet5, SimpleCNN)
    │       ├── dl.models.resnet (resnet18)
    │       ├── dl.models.mlp (MLP)
    │       └── dl.models.linear (LinearRegressionModel, LogisticRegressionModel)
    ├── dl.io.paths (prepare_run_dir)
    ├── dl.io.checkpoints (save_checkpoint, load_checkpoint)
    ├── dl.training.trainer (Trainer)
    │       ├── dl.training.metrics (accuracy, mse, mae)
    │       └── dl.training.callbacks (EarlyStopping, ModelCheckpoint)
    ├── dl.utils.device (resolve_device)
    ├── dl.utils.logging (setup_logging)
    ├── dl.utils.seed (set_seed)
    └── dl.viz.plot (plot_history)
```

---

## 四、快速开始

### 4.1 环境

```bash
cd Python/src
pip install -r requirements.txt
```

### 4.2 训练 MNIST + LeNet

```bash
cd Python/src
python train.py --config configs/mnist_lenet.yaml
```

输出将保存在 `outputs/mnist-lenet-{时间戳}/`。

### 4.3 推理与可视化

```bash
python infer.py --config configs/mnist_lenet.yaml --checkpoint outputs/mnist-lenet-20260211-145945/checkpoints/best.pt --n 16
```

### 4.4 运行测试

```bash
cd Python/src
pytest tests/ -v
```

---

## 五、配置说明

### 5.1 顶层配置

| 字段 | 说明 |
|------|------|
| `project` | 项目名称 |
| `run_name` | 运行名称，用于生成输出目录 |
| `output_dir` | 输出根目录 |
| `task` | 任务类型：classification / regression |

### 5.2 数据配置 (data)

| 字段 | 说明 |
|------|------|
| `name` | mnist / cifar10 / csv_regression / csv_classification / rfid |
| `data_dir` | 数据根目录 |
| `batch_size` | 批大小 |
| `num_workers` | DataLoader 工作进程数 |
| `val_split` | 验证集比例 (0~1) |
| `image_size` | 图像 resize 尺寸（MNIST/CIFAR10） |
| `csv_path` | CSV 文件路径（表格数据） |
| `target_col` | 目标列名 |
| `feature_cols` | 特征列名列表（可选，默认除 target 外所有列） |
| `rfid_dir` | RFID 数据目录（可选，默认 `{data_dir}/RFID`） |
| `rfid_glob` | RFID 文件匹配（默认 `*.csv`） |
| `rfid_label_mode` | RFID 标签编码：observed（仅对存在文件做 0..K-1） / grid（固定网格） |
| `rfid_grid_size` | grid 模式下坐标上限（默认 50，对应 0..50 共 51 个取值） |
| `rfid_window_size` | RFID 时间窗口长度（步数）。1 表示逐行样本；>1 时将在同一坐标文件内按时间顺序滑动窗口构造 `[B,T,F]` 序列样本（当前仅在 `model.name=lstm` 时支持） |

### 5.3 模型配置 (model)

| 字段 | 说明 |
|------|------|
| `name` | lenet / simplecnn / resnet18 / mlp / linear_regression / logistic_regression / lstm |
| `num_classes` | 分类类别数 |
| `in_features` | 输入特征维度（表格模型，可省略由数据推断） |
| `hidden_sizes` | MLP 隐藏层维度列表 |
| `dropout` | Dropout 比例（MLP、LSTM 的全局 dropout） |
| `rnn_hidden_size` | LSTM 隐层维度（默认 64） |
| `rnn_num_layers` | LSTM 堆叠层数（默认 1） |
| `rnn_bidirectional` | LSTM 是否使用双向（默认 false） |

### 5.4 优化器配置 (optim)

| 字段 | 说明 |
|------|------|
| `name` | adam / sgd |
| `lr` | 学习率 |
| `weight_decay` | 权重衰减 |
| `momentum` | SGD 动量 |

### 5.5 训练配置 (train)

| 字段 | 说明 |
|------|------|
| `epochs` | 训练轮数 |
| `seed` | 随机种子 |
| `device` | auto / cuda / cpu |
| `amp` | 是否使用混合精度 |
| `grad_clip_norm` | 梯度裁剪范数（可选） |
| `log_every` | 每隔多少 step 打印一次 |
| `eval_every` | 每隔多少 epoch 验证一次 |
| `early_stopping_patience` | 早停 patience（可选） |

---

## 六、支持的数据集与模型

| 数据集 | 任务 | 推荐模型 |
|--------|------|----------|
| MNIST | 分类 | lenet, mlp |
| CIFAR10 | 分类 | resnet18, simplecnn |
| csv_regression | 回归 | linear_regression, mlp |
| csv_classification | 分类 | logistic_regression, mlp |
| RFID (x_y.csv) | 分类 | mlp, logistic_regression, lstm |

---

## 八、RFID + LSTM 示例

### 8.1 配置文件

已经提供一个示例配置 `configs/rfid_lstm.yaml`，其核心字段如下：

- **顶层**
  - `task: classification`
- **data**
  - `name: rfid`
  - `data_dir: ./data`（默认会使用 `./data/RFID` 目录）
  - `rfid_label_mode: observed`（按实际存在的坐标文件映射类别）
  - `rfid_window_size: 5`（示例：使用长度为 5 的滑动时间窗口）
  - `val_split: 0.2`
- **model**
  - `name: lstm`
  - `rnn_hidden_size: 64`
  - `rnn_num_layers: 1`
  - `rnn_bidirectional: false`
  - `dropout: 0.1`

当 `rfid_window_size = 1`（默认）时，RFID 数据加载器会将每一行 RSSI 特征视为一个样本（形状为 `[B, F]`），
LSTM 模型会自动将其视作单步序列（内部转换为 `[B, 1, F]`）。
当 `rfid_window_size > 1` 且 `model.name = lstm` 时，会在同一坐标文件内部按时间顺序滑动窗口，
构造 `[B, T, F]` 形式的序列样本，以便 LSTM 利用时间信息；窗口不会跨越不同坐标文件。

### 8.2 训练命令

```bash
cd Python/src
python train.py --config configs/rfid_lstm.yaml
```

输出目录形如 `outputs/rfid-lstm-cls-{时间戳}/`，结构与前文说明一致。

---

## 七、版本与依赖

- Python 3.10+
- PyTorch
- torchvision
- pandas
- pyyaml
- matplotlib
- tqdm

详见 `Python/src/requirements.txt`。
