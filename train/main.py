#!/usr/bin/env python
# coding: utf-8
import warnings
import argparse
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from utils.misc import set_seed
from utils.config import load_config
from data.datamodule import CamVidDataModule
from models.segmentation.fpn import FPNSegModel
from models.segmentation.unet import UNetSegModel
from train.trainer import create_trainer

torch.set_float32_matmul_precision('medium')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml", help="配置文件路径")
    parser.add_argument("--model", default="fpn", help="模型类型: fpn 或 unet")
    args = parser.parse_args()
    
    config = load_config(args.config)   # 加载配置
    set_seed(config.training.seed)  # 固定种子(好像起不了作用)
    datamodule = CamVidDataModule(config)   # 初始化数据模块
    
    # 初始化模型
    if args.model == "fpn":
        model = FPNSegModel(config)
    elif args.model == "unet":
        model = UNetSegModel(config)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    trainer = create_trainer(config)    # 创建训练器

    trainer.fit(model, datamodule=datamodule)   # 启动训练
    trainer.test(model, datamodule=datamodule)  # 测试模型


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()