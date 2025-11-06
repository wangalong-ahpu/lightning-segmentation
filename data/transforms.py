#!/usr/bin/env python
# coding: utf-8
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, self.mean, self.std)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return image, target


def get_train_transform():
    return Compose([
        RandomHorizontalFlip(),
        ToTensor(),
    ])


def get_val_transform():
    return Compose([
        ToTensor(),
    ])