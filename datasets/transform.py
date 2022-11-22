# -*- encoding: utf-8 -*-
import os
import cv2
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RandomDiscreteScale(A.RandomScale):
    def __init__(self, scales, always_apply=False, p=0.5):
        super(RandomDiscreteScale, self).__init__(0, always_apply=always_apply, p=p)
        self.scales = scales

    def get_params(self):
        return {"scale": random.choice(self.scales)}


class RandomFlipOrRotate(object):
    def __call__(self, sample):
        img1, img2, label = sample['img1'], sample['img2'], sample['label']

        rand = random.random()
        if rand < 1 / 6:
            img1 = np.flip(img1, axis=0)
            img2 = np.flip(img2, axis=0)
            label = np.flip(label, axis=0)

        elif rand < 2 / 6:
            img1 = np.flip(img1, axis=1)
            img2 = np.flip(img2, axis=1)
            label = np.flip(label, axis=1)

        elif rand < 3 / 6:
            img1 = np.rot90(img1, k=1)
            img2 = np.rot90(img2, k=1)
            label = np.rot90(label, k=1)

        elif rand < 4 / 6:
            img1 = np.rot90(img1, k=2)
            img2 = np.rot90(img2, k=2)
            label = np.rot90(label, k=2)

        elif rand < 5 / 6:
            img1 = np.rot90(img1, k=3)
            img2 = np.rot90(img2, k=3)
            label = np.rot90(label, k=3)

        return {'img1': np.ascontiguousarray(img1),
                'img2': np.ascontiguousarray(img2),
                'label': np.ascontiguousarray(label)}


def RandomFlipOrRotateImgMask(img, mask):
    rand = random.random()
    if rand < 1 / 2:
        img = np.flip(img, axis=0)
        mask = np.flip(mask, axis=0)
    else:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
    rand = random.random()
    if rand < 1 / 3:
        img = np.rot90(img, k=1)
        mask = np.rot90(mask, k=1)
    elif rand < 2 / 3:
        img = np.rot90(img, k=2)
        mask = np.rot90(mask, k=2)
    else:
        img = np.rot90(img, k=3)
        mask = np.rot90(mask, k=3)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)

