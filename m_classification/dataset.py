'''
coding:utf-8
@Software: PyCharm
@Time: 2023/12/21 23:43
@Author: Aocf
@versionl: 3.
'''

import cv2
import os
import torch
from torch.utils.data import Dataset

class MyData(Dataset):
    """
    加载分类数据集
    """
    def __init__(self, img_path, img_info, num_class=2,
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
        self.img_labels = self.img_info['label'].to_list()
        self.randm_compose = randm_compose
        self.fixed_compose = fixed_compose
        self.num_class = num_class

    def __getitem__(self, idx):
        """
        根据idx获取img及mask
        :param idx:
        :return:
        """
        img_name = self.img_names[idx]
        img = cv2.imread(os.path.join(self.img_path, img_name))
        if img_name.split('.')[1] == 'png':
            img = img.convert('RGB')
        if self.randm_compose is not None:
            args = self.randm_compose(image=img)
            img = args['image']
        if self.fixed_compose is not None:
            img = self.fixed_compose(img)

        # 标签生成 0  --> tensor([1,0,0,0]) 对应位置填1
        target = torch.zeros(self.num_class)
        target[self.img_labels[idx]] = 1
        return img, target

    def __len__(self):
        return len(self.img_names)