'''
coding:utf-8
@Software: PyCharm
@Time: 2023/11/30 22:07
@Author: Aocf
@versionl: 3.
'''

import os
import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import albumentations as abm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from utils.logger import Logger
from .dataset import MyData

"""
初始化日志
"""
logger = Logger(r'../db/cat_log')
header = """epchs  |  lr  |  train_loss  |  valid_loss  |  precis  |  recall"""
log_line = r'{}  |  {}  |  {}  |  {}  |  {}  |  {}  '
logger(header)

"""
初始化随机种子
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

"""
加载数据集
"""
# 参数预设
num_class = 2
img_size = 224
batch_size = 20
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

# 图片预处理
fixed_compose = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4207, 0.4381, 0.4023],
                                                         [0.1831, 0.1678, 0.1600])])
# 图片增强
randm_compose = abm.Compose(
                    [abm.HorizontalFlip(p=0.3),  # 水平翻转
                    abm.VerticalFlip(p=0.3),     # 垂直翻转
                    abm.RandomRotate90(),   # 随机旋转90度
                    abm.PixelDropout(dropout_prob=0.01,
                                     per_channel=False,
                                     drop_value=0,
                                     mask_drop_value=None,
                                     always_apply=False, p=0.1),
                    abm.OneOf([
                        abm.RandomGamma(gamma_limit=(60, 220), eps=None,
                                        always_apply=False, p=0.5),
                        abm.ColorJitter(brightness=0.07, contrast=0.1,
                                        saturation=0.2, hue=0.15,
                                        always_apply=False, p=0.3)])], p=0.3)
info_path = r'db/trian_mask_box.csv'
img_path = r'db/train'
mask_info = pd.read_csv(info_path, usecols=[0, 2, 3])
mask_info = mask_info[mask_info['name'].isin(os.listdir(img_path))]
mask_info.loc[:, 'label'] = mask_info['pcnt_box']
mask_info = mask_info.reset_index(drop=True)
# 根据mask归为2个类别
mask_info.loc[mask_info['label'] == 1, 'label'] = 0
mask_info.loc[mask_info['label'].isin([2, 3, 4]), 'label'] = 1
# mask_info.loc[mask_info['label'].isin([4]), 'label'] = 2


# 图片小批量筛选
ls = mask_info.index.to_list()
np.random.shuffle(ls)
mask_info = mask_info.loc[ls[:200], :]
mask_info.reset_index(inplace=True, drop=True)

# 划分训练集测试集
sk_split = 4  # 划分k折
sk_train = 0  # 取第几折划分的训练集验证集
dataset0 = MyData(img_path, mask_info,
                  fixed_compose=fixed_compose,
                  randm_compose=randm_compose)
dataset1 = MyData(img_path, mask_info,
                  fixed_compose=fixed_compose)  # 抽取验证集

skfold = StratifiedKFold(n_splits=sk_split,
                         shuffle=True,
                         random_state=2023)
skfold_split = skfold.split(mask_info.index,
                            mask_info.label)

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
模型加载
"""
# model = models.vgg16_bn(weights=None, num_classes=num_class)
model = models.googlenet(weights=None, num_classes=num_class)
def load_pretrained(pre_dict, net_dict):
    """
    加载预训练模型
    :param pre_dict: 预训练参数
    :param net_dict: 模型参数
    :return:
    """
    for k, v in pre_dict.items():
        if k in net_dict.keys() and v.shape == net_dict[k].shape:
            net_dict[k] = v
    return net_dict
## TODO 这种加载方式导致检测效果下降了很多？寻找原因？
pre_model_path = None
save_format = r'googlecat{}_.pth'
pre_model_path = r'googlecat10_.pth'
if pre_model_path is not None:
    pre_dict = torch.load(pre_model_path)
    net_dict = model.state_dict()
    model.load_state_dict(load_pretrained(pre_dict, net_dict))
model = model.to(device)

"""
搭建损失函数
"""
class Focal_Loss(nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps = 1e-7
        loss_1 = -1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
        loss_0 = -1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)
# bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.8)) # 正:负 = 4:1
# loss_fun = nn.CrossEntropyLoss() # TTA的sigmoid为False
loss_fun = nn.BCEWithLogitsLoss() # TTA的sigmoid为True
loss_fun.to(device)

"""
TTA + 验证指标
"""
@torch.no_grad()
def TTA(imgs, model, sigmoid=False):
    """
    在测试模块对一个batch进行多图像增强
    对同一个图获得多个预测[mask1, mask2, mask3, ...]
    融合：mask = (mask1 + mask2 + mask3 + ...) / n
    图像增强：水平翻转、竖直翻转、 旋转、对比度灰度
    """
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

def get_validate_eval(model, dataloader_val, loss_fn):
    """
    计算模型在验证集上评价指标、损失
    :param model:
    :param dataloader_val:
    :param loss_fn:
    :return:
    """
    loss_val = 0
    model.eval()
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader_val):
            targets = targets.float()
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            outputs_tta = TTA(imgs, model)

            loss = loss_fn(outputs, targets)
            loss_val = loss_val + loss

            # 获取标签
            rows = torch.arange(0, targets.shape[0], dtype=torch.long).cuda()
            pred = torch.zeros_like(targets)
            pred[rows, outputs.argmax(1)] = 1
            pred_tta = torch.zeros_like(targets)
            pred_tta[rows, outputs_tta.argmax(1)] = 1
            if i == 0:
                pred_matrix = pred.clone()
                pred_tta_matrix = pred_tta.clone()
                target_matrix = targets.clone()
            else:
                pred_matrix = torch.cat([pred_matrix, pred], dim=0)
                pred_tta_matrix = torch.cat([pred_tta_matrix, pred_tta], dim=0)
                target_matrix = torch.cat([target_matrix, targets], dim=0)
        # 计算各个类别的precision recall
        matric = get_metric(target_matrix, pred_matrix)
        matric_tta = get_metric(target_matrix, pred_tta_matrix)
    return loss_val, matric, matric_tta

# 定义评估指标
def get_metric(matrix_true, matrix_pred):
    """
    统计每个类别的各个指标, 查全率查准率
    :param matrix_true:
    :param matrix_pred:
    :return:
    """
    class_nums = matrix_true.shape[-1]
    matrcs = {}
    for i in range(class_nums):
        target = matrix_true[:, i]
        pred = matrix_pred[:, i]
        c = sum(target * pred)
        t = sum(target)
        p = sum(pred)
        if t != 0:
            prcs = c / t
            prcs = prcs.item()
        else:
            if p == 0:
                prcs = 1
            else:
                prcs = 0

        if p != 0:
            recall = c / p
            recall = recall.item()
        else:
            if t == 0:
                recall = 1
            else:
                recall = 0
        matrcs[i] = [prcs, recall]
    return matrcs

"""
检查模型载入效果
"""
# model.eval()
# with torch.no_grad():
#     for i, (imgs, targets) in enumerate(train_loader):
#         targets = targets.float()
#         imgs, targets = imgs.to(device), targets.to(device)
#         outputs = model(imgs)
#         rows = torch.arange(0, targets.shape[0], dtype=torch.long).cuda()
#         pred = torch.zeros_like(targets)
#         pred[rows, outputs.argmax(1)] = 1
#         if i == 0:
#             pred_matrix = pred.clone()
#             target_matrix = targets.clone()
#         else:
#             pred_matrix = torch.cat([pred_matrix, pred], dim=0)
#             target_matrix = torch.cat([target_matrix, targets], dim=0)
#     # 计算各个类别的precision recall
#     matric = get_metric(target_matrix, pred_matrix)

"""
指定义优化器
"""
lr = 1e-5
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
    for (imgs, targets) in train_loader:
        targets = targets.float()
        imgs, targets = imgs.to(device), targets.to(device)

        ## 计算损失
        #outputs = model(imgs)
        #loss = loss_fun(outputs, targets)
        #loss_train = loss_train + loss.item()

        #### google_net loss 有三部分
        outputs, aux_logits2, aux_logits1 = model(imgs)
        loss0 = loss_fun(outputs, targets)
        loss1 = loss_fun(aux_logits1, targets)
        loss2 = loss_fun(aux_logits2, targets)
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss_train = loss_train + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_val, matric, matric_tta = get_validate_eval(model, valid_loader, loss_fun)
    print(matric)
    print(matric_tta)
    p = np.mean([x[0] for x in matric.values()])
    r = np.mean([x[1] for x in matric.values()])

    logger(log_line.format(epcoh,
                           optimizer.state_dict()['param_groups'][0]['lr'],
                           loss_train,
                           loss_val,
                           p,
                           r))
    scheduler.step(r)  # lr优化
    if epcoh % 10 == 0:
        torch.save(model.state_dict(), save_format.format(epcoh))

