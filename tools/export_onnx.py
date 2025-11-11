#!/usr/bin/env python
# coding: utf-8
"""
将训练好的模型导出为ONNX格式
"""
import os
import sys
import argparse
import torch
from utils.config import load_config
from models.segmentation.fpn import FPNSegModel
from models.segmentation.unet import UNetSegModel

def export_to_onnx(model, input_shape, output_path, device, opset_version=18, external_data=True):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model: PyTorch模型实例
        input_shape: 输入张量的形状 (batch_size, channels, height, width)
        output_path: ONNX模型保存路径
        device: 运行设备
        opset_version: ONNX操作集版本
        external_data: 是否将大权重保存到外部文件（默认为True，避免2GB限制）
    """
    # 设置模型为评估模式
    model.eval()
    model.to(device)
    
    # 创建示例输入张量
    dummy_input = torch.randn(input_shape, device=device)
    
    # 设置导出参数
    export_params = {
        'model': model,
        'args': dummy_input,
        'f': output_path,
        'export_params': True,
        'opset_version': opset_version,
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        'external_data': external_data  # 控制是否使用外部数据格式
    }
    
    # 导出ONNX模型
    torch.onnx.export(**export_params)
    
    print(f"模型已成功导出到: {output_path}")
    
    # 检查是否产生了外部数据文件
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        size = os.path.getsize(data_file)
        print(f"注意：权重数据已保存到外部文件 {data_file}，大小为 {size / (1024*1024):.1f}MB")

def main():
    parser = argparse.ArgumentParser(description='导出模型为ONNX格式')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='配置文件路径')
    parser.add_argument('--model_type', type=str, default='fpn', choices=['fpn', 'unet'], help='模型类型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--height', type=int, default=256, help='输入图像高度')
    parser.add_argument('--width', type=int, default=256, help='输入图像宽度')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset版本')
    parser.add_argument('--external_data', action='store_true', help='使用外部数据格式')
    
    args = parser.parse_args()
    
    # 默认不使用外部数据格式，除非明确指定
    use_external_data = args.external_data
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    if args.model_type == 'fpn':
        model = FPNSegModel(config)
    elif args.model_type == 'unet':
        model = UNetSegModel(config)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 加载检查点
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"成功加载检查点: {args.checkpoint}")
        
        # 根据检查点路径确定输出路径
        output_path = args.checkpoint.replace('.ckpt', '.onnx')
    else:
        raise ValueError("必须提供检查点路径")
    
    # 设置输入形状 (batch_size, channels, height, width)
    input_shape = (1, config.model.in_channels, args.height, args.width)
    
    # 导出ONNX模型
    export_to_onnx(model, input_shape, output_path, device, args.opset, use_external_data)
    
    print("ONNX模型导出完成!")

if __name__ == '__main__':
    main()

    # # 导出FPN模型
    # python tools/export_onnx.py --checkpoint ./outputs/checkpoints/best-model-epoch=149-valid_dataset_iou=0.9731.ckpt

    # # 导出UNet模型
    # python tools/export_onnx.py --model_type unet --config configs/model/unet.yaml --checkpoint ./outputs/checkpoints/best-model-epoch=149-valid_dataset_iou=0.9731.ckpt

    # # 使用外部数据格式导出
    # python tools/export_onnx.py --model_type unet --config configs/model/unet.yaml --checkpoint ./outputs/checkpoints/best-model-epoch=149-valid_dataset_iou=0.9731.ckpt --external_data