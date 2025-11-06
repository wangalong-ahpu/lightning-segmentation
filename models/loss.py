#!/usr/bin/env python
# coding: utf-8
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_loss_function(loss_name, **kwargs):
    """
    获取损失函数
    """
    if loss_name == "dice":
        return smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == "focal":
        return smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
    elif loss_name == "dice_ce":
        dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        ce = nn.CrossEntropyLoss()
        return lambda logits, targets: dice(logits, targets) + ce(logits, targets)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")