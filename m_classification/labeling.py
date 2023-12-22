'''
coding:utf-8
@Software: PyCharm
@Time: 2023/12/21 23:02
@Author: Aocf
@versionl: 3.
'''
from utils.rel import *
import pandas as pd
from torchvision import transforms
import os
import cv2
import torch

# 标签统计+通道均值方差统计
"""
统计标签的比例
统计通道的均值方差
"""
info_path = r'db/train_mask.csv'
img_path = r'db/train'
mask_info = pd.read_csv(info_path,
                        sep='\t',
                        names=['name', 'mask'])
mask_info = mask_info[(mask_info['name'].isin(os.listdir(img_path)))
                      &(~mask_info['mask'].isna())]
mask_info = mask_info.reset_index(drop=True)

fixed_compose = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

def get_mean_std(img):
    img = fixed_compose(img)
    return img.mean([1,2]), img.std([1,2])
u_tol = torch.zeros(3)
std_tol = torch.zeros(3)

for img_name in mask_info['name']:
    img = cv2.imread(os.path.join(img_path, img_name))
    u, std = get_mean_std(img)
    u_tol = u_tol + u
    std_tol = std_tol + std
print(f'通道均值{u_tol / mask_info.shape[0]}')
print(f'通道方差{std_tol / mask_info.shape[0]}')

# 统计每个样本正负比例的分布
mask_info.loc[:, 'pcnt'] = 0
for i in mask_info.index:
    label1 = np.sum(rle_decode(mask_info.loc[i, 'mask']) == 1)
    label0 = np.sum(rle_decode(mask_info.loc[i, 'mask']) == 0)
    mask_info.loc[i, 'pcnt'] = round(label1 / (label0 + label1), 4)
# 标签划分
box = [-0.1, 0.05, 0.15, 0.25, 0.35, 1]
mask_info.loc[:, 'pcnt_box'] = pd.cut(mask_info['pcnt'],
                                      bins=box,
                                      labels=[x for x in range(len(box) - 1)])
mask_info.to_csv(r'db/trian_mask_box.csv', index=False)
