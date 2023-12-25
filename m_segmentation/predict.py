'''
coding:utf-8
@Software: PyCharm
@Time: 2023/11/20 15:02
@Author: Aocf
@versionl: 3.
'''

import os
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
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
03: 模型融合，多折求平均，多模型投票
{m1:sigmoid(avg[r_k0, r_k2,...]), m2:sigmoid(avg[r_k0, r_k2,...]), ...}
多模型投票模型数量>=3
"""

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
class TestData(Dataset):
    """
    批量加载图片及mask
    mask解码 --> 对应矩阵式
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

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    # pixels = im.flatten(order='F')  # 列展平
    pixels = im.cpu().flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

@torch.no_grad()
def _get_model(net='unet2', encoder='resnet34'):
    if net == 'unet2':
        model = smp.UnetPlusPlus(encoder_name=encoder,
                                 encoder_weights=None,
                                 in_channels=3,
                                 classes=1)
    elif net == 'unet':
        model = smp.Unet(encoder_name=encoder,
                         encoder_weights=None,
                         in_channels=3,
                         classes=1)
    elif net == 'deeplabv3':
        model = smp.DeepLabV3Plus(encoder_name=encoder,
                                encoder_weights=None,
                                in_channels=3,
                                classes=1)
    else:
        raise '模型载入失败， 需重新定义'
    return model


@torch.no_grad()
def get_kfole_medels(pth_path, net='unet2', encoder='resnet31'):
    """
    取一个模型对应的多折训练子模型
    :param net:
    :param encoder:
    :return: [m_k0, m_k2,...]
    """
    key = net + '_' + encoder
    pths = os.listdir(pth_path)
    pths = [x for x in pths if x.startswith(key)]
    if len(pths) == 0:
        return None
    model_ls = []
    for pth in pths:
        model = _get_model(net, encoder)
        model.load_state_dict(torch.load(os.path.join(pth_path,
                                                      pth)))
        model = model.to(device)
        model_ls.append(model)
    return model_ls

def TTA(imgs, model):
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
        mask1 = torch.sigmoid(model(imgs))
        # 翻转
        mask2 = torch.sigmoid(vflip(model(vflip(imgs))))
        mask3 = torch.sigmoid(hflip(model(hflip(imgs))))
        # 旋转
        mask4 = torch.sigmoid(rota_90ab(model(rota_90(imgs))))
        mask5 = torch.sigmoid(rota_90(model(rota_90ab(imgs))))
        # 灰度对比度
        mask6 = torch.sigmoid(model(jiter(imgs)))
    return (mask1 + mask2 + mask3 + mask4 + mask5 + mask6) / 6

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
    masks = []
    for model in models:
        model.eval()
        masks.append(TTA(img, model))
    masks = torch.concat(masks)
    return masks.sum(0) / len(models)

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
    for net in ['unet2', 'unet']:
        for encoder in ['resnet34', 'efficientnet-b4']:
            m = get_kfole_medels(pth_path, net, encoder)
            if m is not None:
                model_dict[net + '_' + encoder] = m

    # 获取测试数据
    fixed_compose = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4301, 0.4684, 0.4640],
                                                             [1.0833e-5, 1.1080e-5, 1.0125e-5])])
    df = pd.read_csv(r'db/test_a_samplesubmit.csv',
                     sep='\t',
                     names=['img_name', 'rle'])
    df.loc[:, 'rle'] = [''] * df.shape[0]
    df.reset_index(inplace=True, drop=True)
    test_data = TestData(img_path=r'db/test_a',
                         img_names=df['img_name'].to_list(),
                         fixed_compose=fixed_compose)
    # 扫描检测
    for idx, img in enumerate(test_data):
        print(idx, '/', df.shape[0])
        img = img.to(device)
        masks = []
        for k, models in model_dict.items():
            masks.append(get_predict(img, models))
        mask = bagaing(masks)
        df.loc[idx, 'rle'] = rle_encode(mask)
    df.to_csv(r'db/test_pred.csv')








