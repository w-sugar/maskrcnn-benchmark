# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose3d(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, target):
        for t in self.transforms:
            images, target = t(images, target)
        return images, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize3d(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, target=None):
        size = images[0].size
        size_pro = self.get_size(size)
        images = [F.resize(image, size_pro) for image in images]
        if target is None:
            return images
        target = target.resize(images[0].size)
        return images, target


class RandomHorizontalFlip3d(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target):
        if random.random() < self.prob:
            images = [F.hflip(image) for image in images]
            target = target.transpose(0)
        return images, target

class RandomVerticalFlip3d(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target):
        if random.random() < self.prob:
            images = [F.vflip(image) for image in images]
            target = target.transpose(1)
        return images, target

class ColorJitter3d(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, images, target):
        images = [self.color_jitter(image) for image in images]
        return images, target


class ToTensor3d(object):
    def __call__(self, images, target):
        images = [F.to_tensor(image) for image in images]
        return images, target


class Normalize3d(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255
        # self.to_bgr255 = False

    def __call__(self, images, target=None):
        if self.to_bgr255:
            images = [image[[2, 1, 0]] * 255 for image in images]
        images = [F.normalize(image, mean=self.mean, std=self.std) for image in images]
        if target is None:
            return images
        return images, target
