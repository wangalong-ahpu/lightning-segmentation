#!/usr/bin/env python
# coding: utf-8
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def create_trainer(config):
    """
    创建训练器
    """
    # 日志配置
    logger = TensorBoardLogger(
        save_dir=config.outputs.log_dir,
        name="segmentation"
    )

    # checkpoint回调（保存最优模型）
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.outputs.ckpt_dir,
        monitor="valid_dataset_iou",
        mode="max",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{valid_dataset_iou:.4f}"
    )

    # 初始化Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="gpu" if config.training.device == "cuda" else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",  # 多GPU时启用DDP策略
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        # deterministic=False,  # 修改为False以解决histc操作的确定性问题
    )
    
    return trainer