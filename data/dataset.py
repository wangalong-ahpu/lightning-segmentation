#!/usr/bin/env python
# coding: utf-8
import os
import torch
import torchvision.io as io
from torch.utils.data import Dataset


class CamVidDataset(Dataset):
    """CamVid Dataset"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        # 使用torchvision.io读取真实图像和掩码
        image = io.read_image(image_path)  # C,H,W格式
        mask = io.read_image(mask_path)    # C,H,W格式
        
        # 转换图像数据类型为float并归一化到[0,1]范围
        image = image.float() / 255.0
        
        # 调整图像和掩码尺寸以满足模型输入要求
        # 使用插值调整图像大小
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        # 使用最近邻插值调整掩码大小
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0).long()
        
        # 处理掩码格式，确保它是单通道的
        if mask.shape[0] == 3:  # 如果掩码有3个通道
            mask = mask[0]      # 只取第一个通道
        elif mask.shape[0] == 1:  # 如果掩码只有1个通道
            mask = mask.squeeze(0)  # 移除通道维度
            
        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)
            
        return image, mask