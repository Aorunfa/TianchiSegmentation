'''
coding:utf-8
@Software: PyCharm
@Time: 2023/11/21 16:34
@Author: Aocf
@versionl: 3.
'''

from predict import *

"""
对训练集绩效预测
抓取预测dice<0.8的样本定义为难分样本
重采样好分样本与难分样本重新训练模型
"""
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Retnp.concatenate([[0], pixels, [0]])urns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

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
    dice = (2 * overlap) / (union + 0.001)
    return dice


# 获取模型
pth_path = r'pths'
model_dict = {}
for net in ['unet2', 'unet', 'deeplabv3']:
    for encoder in ['resnet34', 'efficientnet-b4']:
        m = get_kfole_medels(pth_path, net, encoder)
        if m is not None:
            model_dict[net + '_' + encoder] = m

# 获取测试数据
fixed_compose = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4301, 0.4684, 0.4640],
                                                         [1.0833e-5, 1.1080e-5, 1.0125e-5])])
df = pd.read_csv('db/train_mask.csv',
                        sep='\t',
                        names=['img_name', 'mask'])
df = df[(df['img_name'].isin(os.listdir('db/train')))
                       &(~df['mask'].isna())]
df = df.reset_index(drop=True)
df.loc[:, 'dic'] = [0] * df.shape[0]

train_data = TestData(img_path=r'db/train',
                      img_names=df['img_name'].to_list(),
                      fixed_compose=fixed_compose)
# 扫描检测
for idx, img in enumerate(train_data):
    print(idx, '/', df.shape[0])
    masks = []
    for models in model_dict:
        masks.append(get_predict(img, models))
    output = bagaing(masks)
    mask = rle_decode(df.loc[idx, 'mask'])
    df.loc[idx, 'dic'] = dic_coef(output, mask)

df.to_csv(r'db\df_train_dic.csv', encoding='utf_8_sig')
