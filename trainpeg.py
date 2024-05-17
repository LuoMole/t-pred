import os
import pickle as pk
import time
import numpy as np
import pandas as pd
import dt_process as dt
import torch

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,  f1_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score
from torch import nn
from torch.optim import Adam, SGD, Adagrad, Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import GRU32
from utils import get_dirlist, draw_roc, draw_plot
from argparse import ArgumentParser




# args parser
parser = ArgumentParser(description='Specifying Input Parameters')
parser.add_argument('-e', '--epochs', default=20, type=int, help="Specify the number of epochs")
parser.add_argument('-lr', '--learning_rate', default=0.00001, type=int, help="Specify the learning rate")
parser.add_argument('-b', '--batchsize', default=32,  type=int, help="Specify the batchsize")
parser.add_argument('-d', '--device', default='GPU', help="Specify the device")
parser.add_argument('-CodingfilePath', default=r'C:\Users\86875\Desktop\learning\t-pred\data\blosum50.pk', help="CodingfilePath")
parser.add_argument('-val', '--ValPath', default=r'C:\Users\86875\Desktop\learning\t-pred\code\data\val_sampled.csv', help="ValPath")
parser.add_argument('--datafolder', default=r'C:\Users\86875\Desktop\learning\t-pred\data', help="datafolder")
parser.add_argument('--datastoragefolder', default=r'C:\Users\86875\Desktop\learning\t-pred\code\data', help="datastoragefolder")
parser.add_argument('--patience4earlystop', default=21, type=int, help="patience for earlystop")
parser.add_argument('--patience4lr', default=10, type=int, help="patience for lr scheduler")
args = parser.parse_args()

# log
# 已有改进：bn层  改变了卷积核尺寸（18+） 增加了深度 延长时间 增加模型深度 替换编码矩阵 fliter2*n blosum62 sgdm 宽度(cdr+)
# running:
# 下一步改进：    深度（   全连接层    与卷积层）    gru，
# 提示报错信息

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


start_time = time.time()
epoch = args.epochs
lr_initial = args.learning_rate
batch_size = args.batchsize

device = torch.device('cuda')


aa_tcr = pk.load(open(r'C:\Users\86875\Desktop\learning\t-pred\code\heng\emb\aa_tcr.pk', 'rb'))
aa_ep = pk.load(open(r'C:\Users\86875\Desktop\learning\t-pred\code\heng\emb\aa_tcr.pk', 'rb'))

data_folder = args.datafolder
data_storeage_folder = args.datastoragefolder

best_val_loss = float('inf')


class MingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        m_cdr3 = self.df.iloc[idx]['CDR3']
        m_ep = self.df.iloc[idx]['Epitope']
        m_label = self.df.iloc[idx]['Label']
        return m_cdr3, m_ep, m_label


loss = nn.CrossEntropyLoss()
loss = loss.to(device)

# early stop
stand = args.patience4earlystop
no_improvement_count = 0


# val_data
encoded_val = pd.DataFrame()
val_data = pd.read_csv(args.ValPath)
encoded_val = dt.encode_all(val_data, encoded_val, aa_tcr, aa_ep)
dataset_val = MingDataset(encoded_val)
dataloader_val = DataLoader(dataset_val, batch_size, shuffle=True, drop_last=True)

# train_data
print('开始采样')
train_filenamelist = get_dirlist(data_folder, 'train')
train_filenamelist_sampled = get_dirlist(data_storeage_folder, 'train', 'sampled')


for filename in train_filenamelist:
    if len(train_filenamelist_sampled) == 0:

        print('读取训练文件+采样：', filename)
        # print('开始采样')
        data_sampled = dt.epitope_sample(os.path.join(data_folder, filename), filename.replace('.csv', '_sampled.csv'))
        print('采样完成')
        train_filenamelist_sampled = get_dirlist(data_storeage_folder, 'train', 'sampled')
    else:
        print('读取采样后训练文件')
        break

encoded_file_train_all = pd.DataFrame()

for filename in train_filenamelist_sampled:
    encoded_file_train = pd.DataFrame()
    print('编码采样后训练文件：', filename)
    df_train = pd.read_csv(os.path.join(data_storeage_folder, filename))
    encoded_file_train = dt.encode_all(df_train, encoded_file_train, aa_tcr, aa_ep)
    encoded_file_train_all = pd.concat([encoded_file_train_all, encoded_file_train], ignore_index=True)

dataset_train = MingDataset(encoded_file_train_all)
dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, drop_last=True)

# 训练
model_tcrep = GRU32()
model_tcrep = model_tcrep.to(device)

# model_path = r'C:\Users\86875\Desktop\learning\t-pred\code\model_save\kernelsize18.pth'
# model_tcrep = torch.torch.load(model_path)

# momentum = 0.9
optm_tcrep = Adam(model_tcrep.parameters(), lr_initial)
lr_scheduler = ReduceLROnPlateau(optm_tcrep, mode='min', factor=0.1, patience=args.patience4lr, verbose=True)

loss_train = []
loss_val = []
precision_val = []
precision_train = []
recall_val = []
recall_train = []
pr_re_auc = []
f1_val = []
f1_train = []
acc_val = []
acc_train = []
auc_val = []
auc_train = []

final_epoch = 1
best_model = None
print('开始训练')
for i in tqdm(range(epoch)):
    model_tcrep.train()
    # print(f'第{i}次循环')
    # print('训练模式')
    time1 = time.time()
    loss_epoch_train = []
    acc_epoch_train = []

    # 训练模式
    for cdr3_train, ep_train, label_train in dataloader_train:
        # 将 cdr3_train 转换为张量
        cdr3_tensors = [torch.stack(sample, dim=0) for sample in cdr3_train]
        # 将 cdr3_tensors 转换为一个张量，添加批次维度
        cdr3 = torch.stack(cdr3_tensors, dim=0).to(torch.float32)
        cdr3 = cdr3.to(device)
        cdr3 = cdr3.permute(2, 1, 0)

        ep_tensors = [torch.stack(sample, dim=0) for sample in ep_train]
        ep = torch.stack(ep_tensors, dim=0).to(torch.float32)
        ep = ep.to(device)
        ep = ep.permute(2, 1, 0)

        label = label_train.clone()
        label = label.unsqueeze(1).to(torch.float32)
        label = label.to(device)

        output = model_tcrep(cdr3, ep)
        los_train = loss(output, label)

        pred_label = (output > 0.5).float()
        pred_cpu_numpy = pred_label.cpu().detach().numpy()
        label_cpu_numpy = label.cpu().detach().numpy()

        acc = accuracy_score(label_cpu_numpy, pred_cpu_numpy)
        acc_epoch_train.append(acc)

        optm_tcrep.zero_grad()
        los_train.backward()
        optm_tcrep.step()
        loss_epoch_train.append(los_train.item())
    loss_train_epoch = np.mean(loss_epoch_train)
    loss_train.append(loss_train_epoch)
    acc_train_epoch = np.mean(acc_epoch_train)
    acc_train.append(acc_train_epoch)

    # time2 = time.time()
    # print(f'\n第{i}个epoch训练用时:{time2 - time1}')
    loss_epoch_val = []
    label_all = []
    pred_all = []
    output_all = []

    model_tcrep.eval()
    # print('测试模式')
    time3 = time.time()
    with torch.no_grad():
        for cdr3_val, ep_val, label_val in dataloader_val:
            # 将 cdr3_train 转换为张量
            cdr3_tensors = [torch.stack(sample, dim=0) for sample in cdr3_val]
            # 将 cdr3_tensors 转换为一个张量，添加批次维度
            cdr3 = torch.stack(cdr3_tensors, dim=0).to(torch.float32)
            cdr3 = cdr3.to(device)
            cdr3 = cdr3.permute(2, 1, 0)

            ep_tensors = [torch.stack(sample, dim=0) for sample in ep_val]
            ep = torch.stack(ep_tensors, dim=0).to(torch.float32)
            ep = ep.to(device)
            ep = ep.permute(2, 1, 0)

            label = label_val.clone().detach()
            label = label.unsqueeze(1).to(torch.float32)
            label = label.to(device)

            out_put = model_tcrep(cdr3, ep)
            los_val = loss(out_put, label)
            loss_epoch_val.append(los_val.item())

            # print(loss_test)
            # 阈值设为0.5
            pred_label = (out_put > 0.5).float()
            # acc需要numpy，先使用detach拷贝数据，再转到cpu上
            pred_cpu_numpy = pred_label.cpu().detach().numpy()
            label_cpu_numpy = label.cpu().detach().numpy()
            output_cpu_numpy = out_put.cpu().detach().numpy()

            label_all.append(label_cpu_numpy)
            pred_all.append(pred_cpu_numpy)
            output_all.append(output_cpu_numpy)

        # time4 = time.time()
        # print(f'测试+绘图用时:{time4 - time3}')
    label_all = np.concatenate(label_all)
    pred_all = np.concatenate(pred_all)
    output_all = np.concatenate(output_all)

    fpr, tpr, _ = roc_curve(label_all, output_all)
    roc_auc = auc(fpr, tpr)
    pr, re, _ = precision_recall_curve(label_all, output_all)
    pr_re_auc = average_precision_score(label_all, output_all)
    f1 = f1_score(label_all, pred_all)
    acc = accuracy_score(label_all, pred_all)
    precision = precision_score(label_all, pred_all)
    recall = recall_score(label_all, pred_all)

    mean_loss_epoch = np.mean(loss_epoch_val)
    loss_val.append(mean_loss_epoch)

    f1_val.append(f1)
    acc_val.append(acc)
    precision_val.append(precision)
    recall_val.append(recall)
    auc_val.append(roc_auc)
    print(f'\ttrain_acc:{acc_train_epoch} val_acc:{acc}')
    lr_scheduler.step(mean_loss_epoch)  # 使用验证集上的损失来更新学习率

    final_epoch = i + 1
    # evaluate for best model
    if mean_loss_epoch < best_val_loss:
        best_val_loss = mean_loss_epoch
        best_model = model_tcrep
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= stand:
        torch.save(best_model, 'model_save/model_tcr_earlystop.pth')
        draw_roc(fpr, tpr, roc_auc, 'val_roc')
        print(f'Early stopping at epoch {i}')
        break

    if i == epoch-1:
        torch.save(best_model, 'model_save/gru32.pth')
        draw_roc(fpr, tpr, roc_auc, 'val_roc')

    time2 = time.time()
    print(f'\n第{i}个epoch用时:{time2 - time1}')


plt.figure()
plt.plot(range(final_epoch), loss_train, color="darkorange", lw=2, label='loss_train')
plt.plot(range(final_epoch), loss_val, color="blue", lw=2, label='loss_val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss_curve_train')
plt.legend(loc='lower right')
plt.savefig('fig/loss_curve_train')
# acc
plt.figure()
plt.plot(range(final_epoch), acc_train, color="darkorange", lw=2, label='acc_train')
plt.plot(range(final_epoch), acc_val, color="blue", lw=2, label='acc_val')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Acc_curve_train')
plt.legend(loc='lower right')
plt.savefig('fig/Acc_curve_train')

draw_plot(range(final_epoch), auc_val, 'auc_val', 'auc_val')
draw_plot(range(final_epoch), f1_val, 'f1_val', 'f1_val')
draw_plot(range(final_epoch), precision_val, 'precision_val', 'precision_val')
draw_plot(range(final_epoch), recall_val, 'recall_val', 'recall_val')
