#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import random


def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可重现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda"):
    """
    获取设备
    """
    return torch.device(device_name if torch.cuda.is_available() else "cpu")