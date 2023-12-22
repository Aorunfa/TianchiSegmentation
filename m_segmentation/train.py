'''
coding:utf-8
@Software: PyCharm
@Time: 2023/10/24 23:47
@Author: Aocf
@versionl: 3.
'''

import os
import torch
import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import logging
import cv2
import pandas as pd
import numpy as np
import albumentations as abm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.logger import Logger
from utils.rel import *
from dataset import MyData
from Net.UHRnet.uhrnet import *
"""
初始化日志
"""
logger = Logger(r'../db/seg_log')
header = """epchs  |  lr  |  train_loss  |  valid_loss  |  dice  |  dice_tta  |  dice_pro"""
log_line = r'{}  |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  '
logger(header)

# 随机种子
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2023)


"""
加载数据集
"""
img_size = 512
batch_size = 2
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
fixed_compose = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.3924, 0.4231, 0.3826],
                                                         [0.1608, 0.1440, 0.1325])])

# 数据增强
randm_compose = abm.Compose(
                    [abm.HorizontalFlip(p=0.3),
                    abm.VerticalFlip(p=0.3),
                    abm.RandomRotate90(),
                    abm.PixelDropout(dropout_prob=0.01,
                                     per_channel=False,
                                     drop_value=0,
                                     mask_drop_value=None,
                                     always_apply=False, p=0.1),
                    abm.OneOf([
                        # abm.Compose([abm.RandomCrop(int(img_size / 2), int(img_size / 2)),
                        #              abm.Resize(img_size, img_size)], p=1),  # 随机裁剪 --> resize原来大小
                        abm.RandomGamma(gamma_limit=(60, 220), eps=None,
                                        always_apply=False, p=0.5),
                        abm.ColorJitter(brightness=0.07, contrast=0.1,
                                        saturation=0.2, hue=0.15,
                                        always_apply=False, p=0.3)])], p=0.3)
# df载入
box_label = [0, 1]
# box_label = [2,3,4]
info_path = r'../db/trian_mask_box.csv'
img_path = r'../db/train'
mask_info = pd.read_csv(info_path)
mask_info = mask_info[(mask_info['name'].isin(os.listdir(img_path)))
                      &(~mask_info['mask'].isna())]
# 筛选大小目标
mask_info = mask_info[mask_info['pcnt_box'].isin(box_label)]
mask_info = mask_info.reset_index(drop=True)

# 采样
ls = mask_info.index.to_list()
np.random.shuffle(ls)
mask_info = mask_info.loc[ls[:2000], :]
mask_info.reset_index(inplace=True, drop=True)

# 分割测试集训练集
sk_split = 4  # 划分k折
sk_train = 0  # 取第几折划分的训练集验证集
dataset0 = MyData(img_path, mask_info,
                 fixed_compose=fixed_compose,
                 randm_compose=randm_compose)
dataset1 = MyData(img_path, mask_info,
                 fixed_compose=fixed_compose) # 抽取验证集

# 按pcnt_box进行层级划分
skfold = StratifiedKFold(n_splits=sk_split,
                          shuffle=True,
                          random_state=2023)
skfold_split = skfold.split(mask_info.index,
                            mask_info.pcnt_box)

for k, (train_idx, valid_idx) in enumerate(skfold_split):
    if k == sk_train:
        train_set, valid_set = Subset(dataset0, train_idx), Subset(dataset1, valid_idx)

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  num_workers=0)
        valid_loader = DataLoader(valid_set,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  num_workers=0)
        break

"""
加载网络模型
"""
model = smp.Unet(encoder_name='efficientnet-b4',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=1)

# model = smp.UnetPlusPlus(encoder_name='resnet34',
#                           encoder_weights='imagenet',
#                           in_channels=3,
#                           classes=1)

# TODO 本地完成URnet的训练
# model = UHRnet(num_classes=1)
# init_weights(model)
pre_model_path = None
# pre_model_path = 'u2resnet38_4.pth'
save_format = r'./pths/0split{}_unet-ef4.pth'

if pre_model_path is not None:
    model.load_state_dict(torch.load(pre_model_path))
model = model.to(device)


"""
自定义损失函数
"""
"""
自定义损失函数
"""
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=0.001, dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        # dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = (2 * tp +  self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


# bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.8)) # 正:负 = 4:1
bce_fn = nn.BCEWithLogitsLoss() # 正:负 = 4:1
dice_fn = SoftDiceLoss()
bce_fn.to(device)
dice_fn.to(device)


# 增加forcal loss
# fcal_fn = smp.losses.FocalLoss(mode='binary', alpha=0.2, gamma=2)
# fcal_fn.to(device)

# nn.MSELoss()
# mse_fn = nn.MSELoss()
# mse_fn.to(device)

def loss_fn(y_pred,
            y_true,
            ratio=0.8,
            log_dic=False):
    """
    损失函数构建
    """
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)

    if log_dic:
        return ratio * bce + (1 - ratio) * (-torch.log2(1 - dice))
    else:
        return ratio * bce + (1 - ratio) * dice


"""
评价指标定义 + tta
"""
def dic_coef(output, mask, thr=0.5):
    """
    评价预测值与真实值
    :param predict:
    :param real:
    :return:
    """
    p = output.reshape(-1)
    t = mask.reshape(-1)
    p = p > thr
    t = t > thr
    union = p.sum() + t.sum()
    overlap = (p * t).sum()
    dice = (2 * overlap + 0.001) / (union + 0.001)
    return dice

@torch.no_grad()
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

"""
open cv后处理
"""
def remove_noise(mask, filter='Gaussian',
                 erode_dilate=True,
                 post_area_threshold=7,
                 post_length_threshold=7):
    """
    去除孤立点、去除孤立团用左像素替换
    :param mask: 预测结果 mask-1 , background-0
    :param post_area_threshold:
    :param post_length_threshold:
    :return:
    """
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask, dtype=np.uint8)
    if erode_dilate:  # 腐蚀膨胀，去边缘毛刺
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if filter == 'Gaussion':
        mask = cv2.GaussianBlur(mask, 3, 0)
    else:
        mask = cv2.medianBlur(mask, 3)

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

"""
验证指标生成
"""
"""
验证指标生成
"""


def get_validate_eval(model, dataloader_val, loss_fn):
    """
    计算模型在验证集上评价指标、损失
    :param model:
    :param dataloader_val:
    :param loss_fn:
    :return:
    """
    model.eval()
    loss_val, dic, dic_tta, dic_prc = 0, 0, 0, 0
    with torch.no_grad():
        for (imgs, masks) in dataloader_val:
            masks = masks.float()
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            outputs_tta = TTA(imgs, model)

            masks = torch.reshape(masks, [batch_size, 1, img_size, img_size])
            loss = loss_fn(outputs, masks)
            loss_val = loss_val + loss

            dic = dic + dic_coef(torch.sigmoid(outputs), masks)
            dic_tta = dic_tta + dic_coef(outputs_tta, masks)

            # 计算后处理后的dice
            outputs_cpu = outputs_tta.detach().cpu()
            ms = []
            for i in range(outputs_cpu.shape[0]):
                m = outputs_cpu[i] >= 0.5
                m = m + 0
                ms.append(torch.tensor(remove_noise(m[0])))
            ms = torch.concat(ms)
            ms = ms.to(device)
            dic_prc = dic_prc + dic_coef(ms, masks)
            # dic_prc = dic_prc + 1

        # 平均dice
        dic = dic / len(dataloader_val)
        dic_tta = dic_tta / len(dataloader_val)
        dic_prc = dic_prc / len(dataloader_val)
    return dic, dic_tta, dic_prc, loss_val

"""
指定义优化器
"""
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=4)  # 根据训练中某些测量值更新lr

"""
网络训练
"""
"""
网络训练
"""
epcohs = 30
for epcoh in range(1, epcohs + 1):
    print(f'-----第{epcoh}轮训练开始------')
    model.train()
    loss_train = 0
    # for (imgs, masks) in tqdm.tqdm(train_loader, desc=f"Training Epoch {epcoh}"):
    for (imgs, masks) in train_loader:

        masks = masks.float()
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)

        # 计算损失
        loss = loss_fn(outputs, masks)
        loss_train = loss_train + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    dic, dic_tta, dic_prc, loss_val = get_validate_eval(model, valid_loader, loss_fn)
    scheduler.step(dic)  # lr优化

    logger(log_line.format(epcoh,
                           optimizer.state_dict()['param_groups'][0]['lr'],
                           loss_train,
                           loss_val,
                           dic,
                           dic_tta,
                           dic_prc))

    """
    设定当前训练模型保存的条件：
    验证集指标有提升 或 接近迭代结束阶段
    """
    if epcoh % 10 == 0:
        torch.save(model.state_dict(), save_format.format(epcoh))