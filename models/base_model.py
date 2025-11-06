#!/usr/bin/env python
# coding: utf-8
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class BaseSegModel(pl.LightningModule):
    """基础分割模型类"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 预分配用于存储每个步骤输出的列表
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # 损失函数
        self.loss_fn = self._build_loss()
        
        # 保存mode用于指标计算
        self.mode = "multiclass"
    
    def forward(self, x):
        # 推理方法
        return self.model(x)
    
    def shared_step(self, batch, stage):
        image = batch[0]
        # 检查 mask 是否为 Tensor 或 tuple
        if isinstance(batch[1], tuple) or isinstance(batch[1], list):
            mask = batch[1][0]  # 如果是 tuple，取第一个元素
        else:
            mask = batch[1]  # 直接使用
            
        # 预测
        logits_mask = self.forward(image)
        
        # 计算损失
        loss = self.loss_fn(logits_mask, mask)
        
        # 对于指标计算，需要将logits转换为预测标签
        # 根据multiclass模式的要求，我们需要在通道维度上取argmax
        if self.mode == "multiclass":
            pred_mask = torch.argmax(logits_mask, dim=1)
        else:
            # 对于其他模式，使用阈值处理
            pred_mask = (logits_mask > 0).long()
        
        # 计算指标
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.config.dataset.num_classes
        )
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def shared_epoch_end(self, outputs, stage):
        # 聚合步骤指标
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        # 每图像 IoU 意味着我们在每个图像上计算交集和并集
        # 然后在整个 epoch 上平均 IoU 分数
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # 数据集 IoU 意味着我们在整个数据集上聚合交集和并集
        # 然后计算 IoU 分数
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)
    
    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # 将每步的指标添加到列表中
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # 清空输出列表
        self.training_step_outputs.clear()
        return
    
    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return
    
    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info
    
    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # 清空输出列表
        self.test_step_outputs.clear()
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config.training.lr))
        
        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=int(self.config.training.T_max), 
                eta_min=float(self.config.training.eta_min)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer
    
    def _build_loss(self):
        """构建损失函数"""
        if self.config.training.loss == "dice":
            return smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        else:
            raise ValueError(f"Unsupported loss function: {self.config.training.loss}")
    
    def _build_model(self):
        """构建模型 - 子类需要实现"""
        raise NotImplementedError