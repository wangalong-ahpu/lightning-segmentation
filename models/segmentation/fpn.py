#!/usr/bin/env python
# coding: utf-8
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.base_model import BaseSegModel
import segmentation_models_pytorch as smp


class FPNSegModel(BaseSegModel):
    """FPN分割模型"""
    
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters(config)
        self.model = self._build_model()
    
    def _build_model(self):
        return smp.create_model(
            "FPN",
            encoder_name=self.config.model.backbone,
            in_channels=self.config.model.in_channels,
            classes=self.config.dataset.num_classes,
            encoder_weights=self.config.model.encoder_weights,
        )