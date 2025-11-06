#!/usr/bin/env python
# coding: utf-8
import sys
import os

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
        
        if stage == "fit" or stage is None:
            # 创建训练和验证数据集
            self.train_dataset = CamVidDataset(
                images_dir=f"{data_root}/train",
                masks_dir=f"{data_root}/trainannot"
            )
            self.val_dataset = CamVidDataset(
                images_dir=f"{data_root}/val",
                masks_dir=f"{data_root}/valannot"
            )
            
        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = CamVidDataset(
                images_dir=f"{data_root}/test",
                masks_dir=f"{data_root}/testannot"
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