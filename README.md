# 基于 PyTorch Lightning 的图像分割训练框架

基于 PyTorch Lightning 的图像分割训练框架，支持多种模型架构（UNet、FPN 等）。

## 项目结构

```
project_name/                # 项目根目录（git 仓库根目录）
├── configs/                 # 配置文件目录（统一管理超参数，避免硬编码）
│   ├── base.yaml            # 基础配置（数据集路径、模型类型、通用超参）
│   ├── train.yaml           # 训练配置（batch_size、lr、epochs、优化器）
│   ├── val.yaml             # 验证配置（评估指标、验证频率）
│   └── model/               # 模型专属配置（不同模型的参数，如 backbone、通道数）
│       ├── unet.yaml
│       └── fpn.yaml
├── data/                    # 数据相关（数据集定义、预处理、数据模块）
│   ├── __init__.py
│   ├── dataset.py           # 自定义 Dataset 类（如之前的分割数据集）
│   ├── datamodule.py        # LightningDataModule 类（封装数据加载全流程）
│   ├── transforms.py        # 数据预处理/增强（训练/验证差异化变换）
│   └── utils.py             # 数据工具函数（如文件列表生成、掩码映射）
├── models/                  # 模型相关（网络结构、损失函数、推理逻辑）
│   ├── __init__.py
│   ├── base_model.py        # 基础模型类（继承 LightningModule，封装通用训练逻辑）
│   ├── segmentation/        # 任务专属模型（按任务拆分，如分类、检测）
│   │   ├── unet.py          # UNet 模型实现
│   │   └── fpn.py           # FPN 模型实现
│   ├── loss.py              # 自定义损失函数（如 DiceLoss、FocalLoss）
│   └── metrics.py           # 自定义评估指标（如 mIoU、PixelAccuracy）
├── train/                   # 训练入口（脚本、训练逻辑封装）
│   ├── __init__.py
│   ├── trainer.py           # 训练器封装（配置 PL Trainer、启动训练）
│   └── main.py              # 项目入口脚本（解析配置、初始化组件、启动训练）
├── utils/                   # 全局工具函数（跨模块复用）
│   ├── __init__.py
│   ├── logger.py            # 日志配置（TensorBoard、WandB 集成）
│   ├── config.py            # 配置文件解析（如 yaml 转字典、参数合并）
│   ├── checkpoint.py        #  checkpoint 工具（加载、保存、断点续训）
│   └── misc.py              # 其他工具（种子固定、设备选择、进度条）
├── tests/                   # 单元测试（可选，大型项目必备）
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_metrics.py
├── outputs/                 # 输出目录（自动生成，存放训练结果）
│   ├── experiments/         # 实验记录（按时间戳/实验名命名）
│   │   ├── 20251106_123456/ # 单实验目录
│   │   │   ├── checkpoints/ # 模型权重
│   │   │   ├── logs/        # 训练日志（TensorBoard/WandB）
│   │   │   ├── config.yaml  # 实验所用配置（备份，可复现）
│   │   │   └── results/     # 评估结果（如混淆矩阵、预测图）
│   └── debug/               # 调试输出（可选）
├── requirements.txt         # 依赖包清单（精确版本，确保可复现）
├── .gitignore               # git 忽略文件（如 outputs、.pyc、环境）
└── README.md                # 项目说明（功能、环境配置、运行命令、目录结构）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

```bash
# 使用默认配置训练FPN模型
python train/main.py --config configs/train.yaml --model fpn

# 使用默认配置训练UNet模型
python train/main.py --config configs/train.yaml --model unet

# 使用特定配置文件训练
python train/main.py --config configs/model/unet.yaml --model unet
```

## 导出ONNX模型

训练完成后，可以将PyTorch模型导出为ONNX格式以便部署：

```bash
# 导出FPN模型为ONNX格式
python export_onnx.py --model_type fpn --config configs/train.yaml --checkpoint path/to/checkpoint.ckpt

# 导出UNet模型为ONNX格式
python export_onnx.py --model_type unet --config configs/model/unet.yaml --checkpoint path/to/checkpoint.ckpt

# 使用特定参数导出（如指定opset版本）
python export_onnx.py --model_type unet --config configs/model/unet.yaml --checkpoint path/to/checkpoint.ckpt --opset 18
```

导出的ONNX模型将保存在与检查点相同的目录下，文件名与检查点相同但后缀为`.onnx`。

## 测试ONNX模型

可以使用以下脚本测试ONNX模型的推理能力：

```bash
# 批量测试文件夹中的图像
python test_onnx.py
```

默认情况下，脚本会处理`datasets/tests`文件夹中的所有图像，并将结果保存到`datasets/test_results`文件夹。

## 使用ONNX模型进行推理

项目还提供了一个封装好的类用于ONNX模型推理：

```python
from segmentation_onnx import SegmentationONNX

# 初始化模型
model = SegmentationONNX(model_path="path/to/model.onnx", input_size=(256, 256))

# 处理单张图像
original_image = cv2.imread("path/to/image.jpg")
prediction = model.infer(original_image)
result_image = model.visualize_result(original_image, prediction)

# 批量处理文件夹中的图像
# 详情请查看 segmentation_onnx.py 中的主函数部分
```

## 配置说明

所有训练参数都在 `configs/` 目录下的 YAML 文件中定义：

- `base.yaml`: 基础配置，包含数据集路径和模型基本参数
- `train.yaml`: 训练相关配置，如学习率、批次大小、优化器等
- `model/*.yaml`: 特定模型的配置文件

## 项目特点

1. **模块化设计**：代码按功能拆分为独立模块，便于维护和扩展
2. **配置驱动**：通过 YAML 配置文件管理所有超参数，避免硬编码
3. **可复现性**：固定随机种子，记录实验配置
4. **易于扩展**：支持添加新模型、数据集和评估指标
5. **日志记录**：集成 TensorBoard 日志记录
6. **模型检查点**：自动保存最佳模型
7. **ONNX支持**：支持将训练好的模型导出为ONNX格式，便于部署

## 支持的模型

- UNet
- FPN (Feature Pyramid Network)

## 致谢
[segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)