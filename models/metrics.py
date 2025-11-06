#!/usr/bin/env python
# coding: utf-8
import torch
import segmentation_models_pytorch as smp


class SegmentationMetrics:
    """分割任务评估指标"""
    
    def __init__(self, num_classes, mode="multiclass"):
        self.num_classes = num_classes
        self.mode = mode
    
    def compute_iou(self, tp, fp, fn, tn, reduction="micro"):
        """
        计算IoU指标
        """
        return smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)
    
    def compute_f1_score(self, tp, fp, fn, tn, reduction="micro"):
        """
        计算F1分数
        """
        return smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)
    
    def compute_accuracy(self, tp, fp, fn, tn, reduction="macro"):
        """
        计算准确率
        """
        return smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction)


def calculate_metrics(preds, targets, num_classes):
    """
    计算分割任务的常用指标
    """
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, targets, mode="multiclass", num_classes=num_classes
    )
    return tp, fp, fn, tn