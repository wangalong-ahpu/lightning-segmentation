#!/usr/bin/env python
# coding: utf-8
"""
测试ONNX模型在本地图片上的推理
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import onnxruntime as ort
from glob import glob

def preprocess_image(image_path, input_size=(256, 256)):
    """
    预处理图像以适配模型输入
    
    Args:
        image_path: 图像路径
        input_size: 模型输入尺寸 (height, width)
        
    Returns:
        preprocessed_image: 预处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换颜色空间 BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整图像尺寸
    image = cv2.resize(image, input_size)
    
    # 归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # 转换为CHW格式并添加批次维度
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # 添加批次维度
    
    return image

def postprocess_output(output, original_size):
    """
    后处理模型输出
    
    Args:
        output: 模型输出
        original_size: 原始图像尺寸 (height, width)
        
    Returns:
        processed_output: 处理后的输出
    """
    # 如果输出是多通道的，获取类别预测
    if len(output.shape) > 2:
        # 对于分割任务，通常在通道维度上取argmax得到类别
        if output.shape[1] > 1:  # 多类别
            output = np.argmax(output, axis=1)
        else:  # 单通道输出
            output = (output > 0.5).astype(np.uint8)
    
    # 移除批次维度
    output = np.squeeze(output, axis=0)
    
    # 调整回原始尺寸
    output = cv2.resize(output.astype(np.uint8), (original_size[1], original_size[0]), 
                        interpolation=cv2.INTER_NEAREST)
    
    return output

def visualize_result(original_image_path, prediction, output_path):
    """
    可视化预测结果
    
    Args:
        original_image_path: 原始图像路径
        prediction: 预测结果
        output_path: 结果保存路径
    """
    # 读取原始图像
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 创建彩色mask
    if len(np.unique(prediction)) <= 2:  # 二值图像
        # 创建绿色mask
        mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        mask[prediction > 0] = [0, 255, 0]  # 绿色
        
        # 叠加mask到原始图像
        overlay = cv2.addWeighted(original_image, 0.7, mask, 0.3, 0)
    else:  # 多类别分割
        # 创建彩色map
        colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        # 为不同类别分配不同颜色
        colors = [
            [0, 0, 0],       # 背景 - 黑色
            [255, 0, 0],     # 类别1 - 红色
            [0, 255, 0],     # 类别2 - 绿色
            [0, 0, 255],     # 类别3 - 蓝色
            [255, 255, 0],   # 类别4 - 青色
            [255, 0, 255],   # 类别5 - 紫色
            [0, 255, 255],   # 类别6 - 黄色
        ]
        
        for i in range(min(len(colors), prediction.max() + 1)):
            colored_mask[prediction == i] = colors[i]
        
        # 叠加mask到原始图像
        overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    
    # 保存结果
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay)
    print(f"结果已保存到: {output_path}")

def test_onnx_model(model_path, image_path, output_path):
    """
    使用ONNX模型进行推理测试
    
    Args:
        model_path: ONNX模型路径
        image_path: 测试图像路径
        output_path: 结果保存路径
    """
    # 创建ONNX运行时会话
    try:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception as e:
        print(f"创建ONNX会话时出错: {e}")
        print("尝试使用CPU执行提供者...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 获取模型输入信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"模型输入形状: {input_shape}")
    
    # 预处理图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")
        
    original_size = original_image.shape[:2]  # (height, width)
    
    input_image = preprocess_image(image_path, (input_shape[3], input_shape[2]))  # (width, height)
    print(f"输入图像形状: {input_image.shape}")
    
    # 运行推理
    try:
        outputs = session.run(None, {input_name: input_image})
        prediction = outputs[0]
        print(f"模型输出形状: {prediction.shape}")
    except Exception as e:
        print(f"模型推理时出错: {e}")
        raise
    
    # 后处理输出
    processed_output = postprocess_output(prediction, original_size)
    
    # 保存预测结果
    result_path = output_path.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
    if not result_path.endswith('_mask.png'):
        result_path = output_path.rsplit('.', 1)[0] + '_mask.png'
        
    cv2.imwrite(result_path, processed_output)
    print(f"预测mask已保存到: {result_path}")
    
    # 可视化结果
    visualize_result(image_path, processed_output, output_path)
    
    return processed_output

def test_folder_onnx_model(model_path, input_folder, output_folder):
    """
    测试文件夹中的所有图像
    
    Args:
        model_path: ONNX模型路径
        input_folder: 输入图像文件夹路径
        output_folder: 输出结果文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    # 获取所有图像文件
    for extension in image_extensions:
        image_paths.extend(glob(os.path.join(input_folder, extension)))
        image_paths.extend(glob(os.path.join(input_folder, extension.upper())))
    
    if not image_paths:
        print(f"在文件夹 {input_folder} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_paths)} 个图像文件")
    
    # 为每个图像运行推理
    for i, image_path in enumerate(image_paths):
        print(f"\n处理图像 ({i+1}/{len(image_paths)}): {os.path.basename(image_path)}")
        
        # 生成输出文件路径
        image_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(image_name)[0]
        output_image_path = os.path.join(output_folder, f"{name_without_ext}_result.png")
        output_mask_path = os.path.join(output_folder, f"{name_without_ext}_mask.png")
        
        try:
            # 运行推理
            test_onnx_model(model_path, image_path, output_image_path)
            print(f"完成处理: {image_name}")
        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {e}")
    
    print(f"\n所有图像处理完成，结果已保存到: {output_folder}")

def main():
    # 默认参数
    model_path = "outputs/checkpoints/best-model-epoch=149-valid_dataset_iou=0.9731.onnx"
    input_folder = "datasets/tests"
    output_folder = "datasets/test_results"
    
    # 测试文件夹中的所有图像
    try:
        test_folder_onnx_model(model_path, input_folder, output_folder)
        print("ONNX模型文件夹测试完成!")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
    # python test_onnx.py