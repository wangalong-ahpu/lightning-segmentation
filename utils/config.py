#!/usr/bin/env python
# coding: utf-8
import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """
    加载配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 转换为DictConfig对象
    config = OmegaConf.create(config)
    
    # 如果有默认配置，合并它们
    if 'defaults' in config:
        for default_config in config.defaults:
            default_path = f"configs/{default_config}"
            default_cfg = load_config(default_path)
            config = OmegaConf.merge(default_cfg, config)
        
        # 删除defaults键，避免重复处理
        del config['defaults']
    
    return config