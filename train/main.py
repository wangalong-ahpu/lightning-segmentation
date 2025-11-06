#!/usr/bin/env python
# coding: utf-8
import argparse
# import torch
# import pytorch_lightning as pl
# from omegaconf import DictConfig, OmegaConf
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 从utils导入工具函数
# from utils.misc import set_seed
from utils.config import load_config

# 从data导入数据模块
from data.datamodule import CamVidDataModule

# 从models导入模型
from models.segmentation.fpn import FPNSegModel
from models.segmentation.unet import UNetSegModel

# 从train导入训练器
from train.trainer import create_trainer


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml", help="配置文件路径")
    parser.add_argument("--model", default="fpn", help="模型类型: fpn 或 unet")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 固定种子（可复现）
    # set_seed(config.training.seed)
    
    # 允许某些操作的非确定性实现，以避免histc等操作的错误
    # torch.use_deterministic_algorithms(False)  # 修改为False以解决histc操作的确定性问题
    
    # 初始化数据模块
    datamodule = CamVidDataModule(config)
    
    # 初始化模型
    if args.model == "fpn":
        model = FPNSegModel(config)
    elif args.model == "unet":
        model = UNetSegModel(config)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    # 创建训练器
    trainer = create_trainer(config)
    
    # 启动训练
    trainer.fit(model, datamodule=datamodule)
    
    # 测试模型
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()