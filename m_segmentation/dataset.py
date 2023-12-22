'''
coding:utf-8
@Software: PyCharm
@Time: 2023/12/22 12:24
@Author: Aocf
@versionl: 3.
'''

import cv2
import os
import torch
from torch.utils.data import Dataset
from utils.rel import *

class MyData(Dataset):
    """
    批量加载图片及mask
    mask解码 --> 对应矩阵式
    """
    def __init__(self, img_path, img_info,
                 fixed_compose=None,
                 randm_compose=None):
        """
        :param img_path: 图片存放根目录 path
        :param img_info: 图片信息 df
        :param randm_compose: 随机处理模块 如Flip
        :param fixed_compose: 固定处理模块，如Totnsor
        """
        super().__init__()
        self.img_path = img_path
        self.img_info = img_info
        self.img_names = self.img_info['name'].to_list()
        self.img_masks = self.img_info['mask'].to_list()
        self.randm_compose = randm_compose
        self.fixed_compose = fixed_compose

    def __getitem__(self, idx):
        """
        根据idx获取img及mask
        :param idx:
        :return:
        """
        img_name = self.img_names[idx]
        mask_rel = self.img_masks[idx]
        img = cv2.imread(os.path.join(self.img_path, img_name))
        mask = rle_decode(mask_rel)  # mask解码
        mask = mask

        if img_name.split('.')[1] == 'png':
            img = img.convert('RGB')
        if self.randm_compose is not None:
            args = self.randm_compose(image=img, mask=mask)
            img, mask = args['image'], args['mask'][None]
        if self.fixed_compose is not None:
            img = self.fixed_compose(img)
        return img, mask

    def __len__(self):
        return len(self.img_names)