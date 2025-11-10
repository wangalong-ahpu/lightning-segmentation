#!/usr/bin/env python
# coding: utf-8
import sys
import os
import random

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import CamVidDataset


class CamVidDataModule(pl.LightningDataModule):
    """CamVid Data Module"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        data_root = self.config.dataset.root
        
        # 获取所有图像和掩码文件
        images_dir = f"{data_root}/images" if os.path.exists(f"{data_root}/images") else data_root
        masks_dir = f"{data_root}/masks" if os.path.exists(f"{data_root}/masks") else data_root.replace("images", "masks")
        
        # 获取所有图像文件名以确定数据集大小
        all_images = sorted(os.listdir(images_dir))
        total_size = len(all_images)
        
        # 根据配置中的比例计算各数据集大小
        train_ratio = self.config.dataset.split.train
        val_ratio = self.config.dataset.split.val
        test_ratio = self.config.dataset.split.test
        
        # 确保比例之和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据集划分比例之和必须为1"
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # 创建索引列表并打乱
        indices = list(range(total_size))
        random.seed(self.config.training.seed)
        random.shuffle(indices)
        
        # 划分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        if stage == "fit" or stage is None:
            # 创建训练和验证数据集
            self.train_dataset = CamVidDataset(
                images_dir=images_dir,
                masks_dir=masks_dir,
                indices=train_indices
            )
            self.val_dataset = CamVidDataset(
                images_dir=images_dir,
                masks_dir=masks_dir,
                indices=val_indices
            )
            
        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = CamVidDataset(
                images_dir=images_dir,
                masks_dir=masks_dir,
                indices=test_indices
            )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers
        )