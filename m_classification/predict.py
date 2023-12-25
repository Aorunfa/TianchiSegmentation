'''
coding:utf-8
@Software: PyCharm
@Time: 2023/12/25 10:12
@Author: Aocf
@versionl: 3.
'''

import os
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision import models
np.random.seed(2023)

"""
预测文件逻辑：
模型pth保存文件夹：一个模型多折训练pth 包含多个模型（>=2）
测试集图片文件夹：所有测试图片img
逻辑：
01: 载入所有模型
{m1:[m_k0, m_k2,...], m2:[m_k0, m_k2,...], ...}
02: 多折模型输出tta结果, no_sigmoid
{m1:[r_k0, r_k2,...], m2:[r_k0, r_k2,...], ...}
03: 模型融合，软投票，硬投
"""
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2023)

num_class = 2
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

class TestData(Dataset):
    """
    批量加载图片
    """
    def __init__(self, img_path, img_names, fixed_compose=None):
        """
        :param img_path: 图片存放根目录 path
        :param fixed_compose: 固定处理模块，如Totnsor
        """
        super(TestData).__init__()
        self.img_path = img_path
        self.img_names = img_names
        self.fixed_compose = fixed_compose

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
        if self.fixed_compose is not None:
            img = self.fixed_compose(img)
        return img

    def __len__(self):
        return len(self.img_names)

@torch.no_grad()
def _get_model(net='googlenet'):
    if net == 'googlenet':
        model = models.googlenet(weights=None, num_classes=num_class)
    elif net == 'vgg':
        model = models.vgg16_bn(weights=None, num_classes=num_class)
    else:
        raise ValueError(f'{net}模型载入失败， 需重新定义')
    return model

@torch.no_grad()
def get_kfole_medels(pth_path, net='google'):
    """
    取一个模型对应的多折训练子模型
    :param net:
    :return: [m_k0, m_k2,...]
    """
    key = net
    pths = os.listdir(pth_path)
    pths = [x for x in pths if x.startswith(key)]
    if len(pths) == 0:
        return None
    model_ls = []
    for pth in pths:
        model = _get_model(net)
        model.load_state_dict(torch.load(os.path.join(pth_path,
                                                      pth)))
        model = model.to(device)
        model_ls.append(model)
    return model_ls

@torch.no_grad()
def TTA(imgs, model, sigmoid=False):
    """
    在测试模块对一个batch进行多图像增强
    对同一个图获得多个预测[mask1, mask2, mask3, ...]
    融合：mask = (mask1 + mask2 + mask3 + ...) / n
    图像增强：水平翻转、竖直翻转、 旋转、对比度灰度
    """
    model.eval()
    with torch.no_grad():
        vflip = transforms.RandomVerticalFlip(p=1)
        hflip = transforms.RandomHorizontalFlip(p=1)
        rota_90 = transforms.RandomRotation((90, 90))
        rota_90ab = transforms.RandomRotation((-90, -90))
        jiter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.15)
        if sigmoid:
            t1 = torch.sigmoid(model(imgs))
            # 翻转
            t2 = torch.sigmoid(model(vflip(imgs)))
            t3 = torch.sigmoid(model(hflip(imgs)))
            # 旋转
            t4 = torch.sigmoid(model(rota_90(imgs)))
            t5 = torch.sigmoid(model(rota_90ab(imgs)))
            # 灰度对比度
            t6 = torch.sigmoid(model(jiter(imgs)))
        else:
            t1 = model(imgs)
            # 翻转
            t2 = model(vflip(imgs))
            t3 = model(hflip(imgs))
            # 旋转
            t4 = model(rota_90(imgs))
            t5 = model(rota_90ab(imgs))
            # 灰度对比度
            t6 = model(jiter(imgs))
    return (t1 + t2 + t3 + t4 + t5 + t6) / 6


@torch.no_grad()
def get_predict(img, models):
    """
    mask预测
    :param img:
    :param model:
    :return:
    """
    # 增加一个通道
    (c, w, h) = img.shape
    img = torch.reshape(img, [1, c, w, h])
    results = []
    for model in models:
        model.eval()
        results.append(TTA(img, model))
    masks = torch.concat(results)
    return masks.sum(0) / len(models)  # 取概率均值

def bagaing(masks, threshold=0.5, method='hard'):
    """
    对多个模型结果进行Bagging, 采用均值概率or投票
    :param masks:
    :return:
    """
    if isinstance(masks, list):
        masks = torch.concat(masks)

    if method == 'hard':  # 软投票
        masks = masks >= threshold
        masks = masks + 0
        mask = masks.sum(0) / masks.shape[0] > 0.501
        return mask + 0
    else:
        # 返回概率均值  # 硬投票
        masks = masks.sum(0) / masks.shape[0] >= threshold
        return masks + 0

if __name__ == '__main__':

    # 获取模型
    pth_path = r'pths'
    model_dict = {}
    for net in ['google', 'vgg']:
        m = get_kfole_medels(pth_path, net)
        if m is not None:
            model_dict[net] = m  # k折模型

    # 获取测试数据
    img_size = 224
    fixed_compose = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4207, 0.4381, 0.4023],
                                                             [0.1831, 0.1678, 0.1600])])
    df = pd.read_csv(r'db/test_a_samplesubmit.csv',
                     sep='\t',
                     names=['img_name', 'rle'])
    df.loc[:, 'class'] = [''] * df.shape[0]
    df.reset_index(inplace=True, drop=True)
    test_data = TestData(img_path=r'db/test_a',
                         img_names=df['img_name'].to_list(),
                         fixed_compose=fixed_compose)
    # 扫描检测
    for idx, img in enumerate(test_data):
        print(idx, '/', df.shape[0])
        img = img.to(device)
        results = []
        for k, models in model_dict.items():
            results.append(get_predict(img, models))
        pred = bagaing(results)
        df.loc[idx, 'class'] = pred
    df.to_csv(r'db/test_class_pred.csv')
