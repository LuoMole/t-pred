import os
import pickle as pk
import time
import numpy as np
import pandas as pd
import dt_process as dt
import torch

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,  f1_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam, SGD, Adagrad, Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import  GRU32
from utils import get_dirlist, draw_roc, draw_plot
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from sklearn.model_selection import KFold


# args parser
parser = ArgumentParser(description='Specifying Input Parameters')
parser.add_argument('-e', '--epochs', default=100, type=int, help="Specify the number of epochs")
parser.add_argument('-lr', '--learning_rate', default=0.0001, type=int, help="Specify the learning rate")
parser.add_argument('-b', '--batchsize', default=32,  type=int, help="Specify the batchsize")
parser.add_argument('-d', '--device', default='GPU', help="Specify the device")
parser.add_argument('-val', '--ValPath', default=r'data-sampled\val_sampled.csv', help="ValPath")
parser.add_argument('--datafolder', default=r'C:\Users\86875\Desktop\learning\t-pred\data', help="datafolder")
parser.add_argument('--datastoragefolder', default='data-sampled', help="datastoragefolder")
parser.add_argument('--patience4earlystop', default=250, type=int, help="patience for earlystop")
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


# aa_tcr = pk.load(open(r'C:\Users\86875\Desktop\learning\t-pred\code\ming\emb\aa_tcr.pk', 'rb'))
# aa_ep = pk.load(open(r'C:\Users\86875\Desktop\learning\t-pred\code\ming\emb\aa_tcr.pk', 'rb'))
aa_tcr = pk.load(open(r'emb\aa_glove_comparison.pk', 'rb'))
aa_ep = pk.load(open(r'emb\aa_glove_comparison.pk', 'rb'))
# for blosum50
data_folder = args.datafolder
data_storeage_folder = args.datastoragefolder

best_val_loss = float('inf')


class MingDataset(Dataset):
    def __init__(self, df, aa_tcr, aa_ep):
        self.df = df
        self.aa_tcr = aa_tcr
        self.aa_ep = aa_ep

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        m_cdr3 = str(self.df.iloc[idx]['CDR3']).upper()
        m_ep = str(self.df.iloc[idx]['Epitope']).upper()
        m_label = self.df.iloc[idx]['Label']
        encoded_cdr3 = dt.encode_cdr(m_cdr3, self.aa_tcr)
        encoded_ep = dt.encode_ep(m_ep, self.aa_ep)
        m_label = torch.tensor(m_label, dtype=torch.float)
        return encoded_cdr3, encoded_ep, m_label


loss = nn.BCELoss()
loss = loss.to(device)

# early stop
stand = args.patience4earlystop
no_improvement_count = 0

data_filenamelist = get_dirlist(data_storeage_folder, 'data','comparison')

file_all = pd.DataFrame()

for filename in data_filenamelist:
    print('编码采样后训练文件：', filename)
    df_train = pd.read_csv(os.path.join(data_storeage_folder, filename))
    file_all = pd.concat([file_all, df_train], ignore_index=True)

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

loss_train = []
loss_val = []
precision_val = []
precision_train = []
recall_val = []
recall_train = []
# pr_re_auc = []
f1_val = []
f1_train = []
acc_val = []
acc_train = []
auc_val = []
auc_train = []
mcc_val = []

for fold, (train_idx, val_idx) in enumerate(kf.split(file_all)):
    print(f'第{fold}折训练开始')
    train_data = file_all.iloc[train_idx]
    val_data = file_all.iloc[val_idx]
    dataset_train = MingDataset(train_data, aa_tcr, aa_ep)
    dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, drop_last=True)

    dataset_val = MingDataset(val_data, aa_tcr, aa_ep)
    dataloader_val = DataLoader(dataset_val, batch_size, shuffle=True, drop_last=True)
    # 训练
    model_tcrep = GRU32()
    model_tcrep = model_tcrep.to(device)

    # model_path = r'C:\Users\86875\Desktop\learning\t-pred\code\model_save\kernelsize18.pth'
    # model_tcrep = torch.torch.load(model_path)

    # momentum = 0.9
    optm_tcrep = Adam(model_tcrep.parameters(), lr_initial)
    lr_scheduler = ReduceLROnPlateau(optm_tcrep, mode='min', factor=0.1, patience=args.patience4lr, verbose=True)



    # final_epoch = 1
    # best_model = None
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
            tcr = cdr3_train.squeeze(dim=-2).to(device)

            ep = ep_train.squeeze(dim=-2).to(device)

            label = torch.Tensor(label_train)
            label = label.to(device)
            onehot_label = torch.eye(2).to(device)[label.long(), :].squeeze()

            output = model_tcrep(tcr, ep)
            output = output.squeeze(dim=-1)
            # print(output)
            los_train = loss(output, onehot_label)

            pred_label = torch.argmax(output, dim=1)
            pred_cpu_numpy = pred_label.cpu().detach().numpy()
            label_cpu_numpy = label.cpu().detach().numpy()

            acc = accuracy_score(label_cpu_numpy, pred_cpu_numpy)
            acc_epoch_train.append(acc)

            optm_tcrep.zero_grad()
            los_train.backward()
            optm_tcrep.step()
            loss_epoch_train.append(los_train.item())
        # loss_train_epoch = np.mean(loss_epoch_train)
        # loss_train.append(loss_train_epoch)
        acc_train_epoch = np.mean(acc_epoch_train)
        # acc_train.append(acc_train_epoch)

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
            for cdr3_val, ep_val, label_val in dataloader_train:
                # 将 cdr3_val 转换为张量
                tcr = cdr3_val.squeeze(dim=-2).to(device)

                ep = ep_val.squeeze(dim=-2).to(device)

                label = torch.Tensor(label_val)
                label = label.to(device)
                onehot_label = torch.eye(2).to(device)[label.long(), :].squeeze()

                out_put = model_tcrep(tcr, ep)
                los_val = loss(out_put, onehot_label)
                loss_epoch_val.append(los_val.item())

                # print(loss_test)
                # 阈值设为0.5
                # pred_label = (out_put > 0.5).float()
                pred_label = torch.argmax(out_put, dim=1)
                # acc需要numpy，先使用detach拷贝数据，再转到cpu上
                pred_cpu_numpy = pred_label.cpu().detach().numpy()
                label_cpu_numpy = label.cpu().detach().numpy()
                output_cpu_numpy = out_put[:, 1].cpu().detach().numpy()

                label_all.append(label_cpu_numpy)
                pred_all.append(pred_cpu_numpy)
                output_all.append(output_cpu_numpy)

            # time4 = time.time()
            # print(f'测试+绘图用时:{time4 - time3}')
        label_all = np.concatenate(label_all)
        pred_all = np.concatenate(pred_all)
        output_all = np.concatenate(output_all)
        mcc = matthews_corrcoef(y_true=label_all, y_pred=pred_all)

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

        if i == epoch-1:
            mcc_val.append(mcc)
            f1_val.append(f1)
            acc_val.append(acc)
            precision_val.append(precision)
            recall_val.append(recall)
            auc_val.append(roc_auc)

        print(f'train_acc:{acc_train_epoch} val_acc:{acc}')
        lr_scheduler.step(mean_loss_epoch)  # 使用验证集上的损失来更新学习率
        print(f'valF1：{f1}')
        print(f'valACC：{acc}')
        print(f'valmcc：{mcc}')
        print(f'valauc：{roc_auc}')
        print(f'valprecision：{precision}')
        print(f'valrecall：{recall}')

        final_epoch = i + 1
        # # evaluate for best model
        # if mean_loss_epoch < best_val_loss:
        #     best_val_loss = mean_loss_epoch
        #     best_model = model_tcrep
        #     no_improvement_count = 0
        # else:
        #     no_improvement_count += 1
        #
        # if no_improvement_count >= stand:
        #     torch.save(best_model, 'model_save/model_tcr_earlystop.pth')
        #     draw_roc(fpr, tpr, roc_auc, 'val_roc')
        #     print(f'Early stopping at epoch {i}')
        #     break
        #
        # if i == epoch-1:
        #     torch.save(best_model, 'model_save/glove_gru.pth')
        #     draw_roc(fpr, tpr, roc_auc, 'val_roc')

        time2 = time.time()
        print(f'\n第{i}个epoch用时:{time2 - time1}')





# plt.figure()
# plt.plot(range(final_epoch), loss_train, color="darkorange", lw=2, label='loss_train')
# plt.plot(range(final_epoch), loss_val, color="blue", lw=2, label='loss_val')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Loss_curve_train')
# plt.legend(loc='lower right')
# plt.savefig('fig/loss_curve_train')
# # acc
# plt.figure()
# plt.plot(range(final_epoch), acc_train, color="darkorange", lw=2, label='acc_train')
# plt.plot(range(final_epoch), acc_val, color="blue", lw=2, label='acc_val')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.title('Acc_curve_train')
# plt.legend(loc='lower right')
# plt.savefig('fig/Acc_curve_train')
#
# draw_plot(range(final_epoch), auc_val, 'auc_val', 'auc_val')
# draw_plot(range(final_epoch), f1_val, 'f1_val', 'f1_val')
# draw_plot(range(final_epoch), precision_val, 'precision_val', 'precision_val')
# draw_plot(range(final_epoch), recall_val, 'recall_val', 'recall_val')
# draw_plot(range(final_epoch), mcc_val, 'recall_val', 'recall_val')
F1 = np.mean(f1_val)
ACC = np.mean(acc_val)
mcc = np.mean(mcc_val)
AUC = np.mean(auc_val)
pre = np.mean(precision_val)
re = np.mean(recall_val)
print(f'valF1_all：{F1}')
print(f'valACC_all：{ACC}')
print(f'valmcc_all：{mcc}')
print(f'valauc_all：{AUC}')
print(f'valprecision_all：{pre}')
print(f'valrecall_all：{re}')
