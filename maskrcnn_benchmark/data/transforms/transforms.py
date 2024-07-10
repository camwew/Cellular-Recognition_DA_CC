# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, image_inpaint, target):
        for t in self.transforms:
            image, image_inpaint, target = t(image, image_inpaint, target)
        return image, image_inpaint, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
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

    def __call__(self, image, image_inpaint, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        image_inpaint = F.resize(image_inpaint, size)
        if target is None:
            return image, image_inpaint
        target = target.resize(image.size)
        return image, image_inpaint, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, image_inpaint, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            image_inpaint = F.hflip(image_inpaint)
            target = target.transpose(0)
        return image, image_inpaint, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, image_inpaint, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            image_inpaint = F.vflip(image_inpaint)
            target = target.transpose(1)
        return image, image_inpaint, target

class ColorJitter(object):
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

    def __call__(self, image, image_inpaint, target):
        image = self.color_jitter(image)
        image_inpaint = self.color_jitter(image_inpaint)
        return image, image_inpaint, target


class ToTensor(object):
    def __call__(self, image, image_inpaint, target):
        return F.to_tensor(image), F.to_tensor(image_inpaint), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, image_inpaint, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
            image_inpaint = image_inpaint[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        image_inpaint = F.normalize(image_inpaint, mean=self.mean, std=self.std)
        if target is None:
            return image, image_inpaint
        return image, image_inpaint, target
