'''
coding:utf-8
@Software: PyCharm
@Time: 2023/11/22 10:17
@Author: Aocf
@versionl: 3.
'''

import os
import cv2
import numpy as np
from PIL import Image
import time
import pickle
import shutil
import random
import copy
import sys
from skimage import morphology
"""
后处理方法：
CRF：条件随机场
小目标移除形成连通区域
中值滤波
"""


def remove_noise(mask, filter='Gaussian',
                 erode_dilate=True,
                 post_area_threshold=25,
                 post_length_threshold=25):
    """
    去除孤立点、去除孤立团用左像素替换
    :param mask: 预测结果 mask-1 , background-0
    :param post_area_threshold:
    :param post_length_threshold:
    :return:
    """
    if erode_dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if filter == 'Gaussion':
        mask = cv2.GaussianBlur(mask, 5, 0)
    else:
        mask = cv2.medianBlur(mask, 5)

    mask_ = mask.copy()
    contours, hierarch = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area <= post_area_threshold or length <= post_length_threshold:
            cnt = contours[i]
            # 左顶点
            location = tuple(cnt[cnt[:, :, 0].argmin()][0])
            class_num = int(mask[location[1], location[0] - 1])
            cv2.drawContours(mask, [cnt], 0, class_num, -1)
    return mask
