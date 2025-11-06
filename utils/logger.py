#!/usr/bin/env python
# coding: utf-8
import logging
from pathlib import Path


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    设置日志记录器
    """
    # 创建日志目录（如果不存在）
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


def log_config(logger, config):
    """
    记录配置信息
    """
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")