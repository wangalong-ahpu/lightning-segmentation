#!/usr/bin/env python
# coding: utf-8
import os


def get_file_list(directory):
    """
    获取目录中所有文件的列表
    """
    return sorted(os.listdir(directory))


def count_files(directory):
    """
    计算目录中文件的数量
    """
    return len(get_file_list(directory))


def create_data_splits(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    创建数据集划分
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    files = get_file_list(data_dir)
    total_files = len(files)
    
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]
    
    return train_files, val_files, test_files