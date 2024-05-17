import os
import pickle as pk
import time
import numpy as np
import pandas as pd
import dt_process as dt
import torch
import os

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,  f1_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score
from torch import nn
from torch.optim import Adam, SGD, Adagrad, Adamax, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import GRU32, CNNemb, GRUemb, FNNemb
from utils import get_dirlist, draw_roc, draw_plot
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

# args parser
parser = ArgumentParser(description='Specifying Input Parameters')
parser.add_argument('-e', '--epochs', default=100, type=int, help="Specify the number of epochs")
parser.add_argument('-lr', '--learning_rate', default=0.001, type=int, help="Specify the learning rate")
parser.add_argument('-b', '--batchsize', default=128,  type=int, help="Specify the batchsize")
parser.add_argument('-d', '--device', default='GPU', help="Specify the device")
parser.add_argument('-val', '--ValPath', default=r'C:\Users\86875\Desktop\learning\t-pred\code\data\val_sampled.csv', help="ValPath")
parser.add_argument('--datafolder', default=r'C:\Users\86875\Desktop\learning\t-pred\data', help="datafolder")
parser.add_argument('--datastoragefolder', default=r'C:\Users\86875\Desktop\learning\t-pred\code\data', help="datastoragefolder")
parser.add_argument('--patience4earlystop', default=21, type=int, help="patience for earlystop")
parser.add_argument('--patience4lr', default=10, type=int, help="patience for lr scheduler")
parser.add_argument('--vocabularysize', default=11, type=int, help="size of vocabulary")
args = parser.parse_args()

# log
# 已有改进：bn层  改变了卷积核尺寸（18+） 增加了深度 延长时间 增加模型深度 替换编码矩阵 fliter2*n blosum62 sgdm 宽度(cdr+)
# running:
# 下一步改进：    深度（   全连接层    与卷积层）    gru，
# 提示报错信息

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致
vocabulary_size = args.vocabularysize

start_time = time.time()
epoch = args.epochs
lr_initial = args.learning_rate
batch_size = args.batchsize

device = torch.device('cuda')



data_folder = args.datafolder
data_storeage_folder = args.datastoragefolder

best_val_loss = float('inf')

aas = ' ACDEFGHIKLMNPQRSTVWY'
aa2idx = {aas[i]: i for i in range(len(aas))}
num_aa = len(aas)


def encode_sequence(sequence, aa2idx):
    encoded_seq = [aa2idx[aa] for aa in sequence]
    return encoded_seq


class MingDataset(Dataset):
    def __init__(self, df, aa2idx):
        self.df = df
        self.aa2idx = aa2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        m_cdr3 = self.df.iloc[idx]['CDR3']
        m_ep = self.df.iloc[idx]['Epitope']
        m_label = self.df.iloc[idx]['Label']
        m_cdr3 = ' ' * (44 - len(m_cdr3)) + m_cdr3
        m_ep = m_ep + ' ' * (15 - len(m_ep))

        # 使用 encode_sequence 函数将 CDR3 和 Epitope 编码为张量
        encoded_cdr3 = torch.tensor(encode_sequence(m_cdr3, self.aa2idx), dtype=torch.long)
        encoded_ep = torch.tensor(encode_sequence(m_ep, self.aa2idx), dtype=torch.long)

        # 将编码后的序列连接起来
        cat = torch.cat((encoded_cdr3, encoded_ep), dim=0)

        return cat, m_label


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    seq_len = [len(s[0]) for s in data]

    sequences = [s[0] for s in data]  # 提取序列
    labels = [s[1] for s in data]  # 提取标签

    # 对序列进行填充
    sequences = pad_sequence(sequences, batch_first=True)
    sequences = sequences.unsqueeze(-1)

    # 将序列 pack
    sequences = pack_padded_sequence(sequences, seq_len, batch_first=True)

    return sequences, labels


loss = nn.BCELoss()
# loss = nn.BCEWithLogitsLoss()
loss = loss.to(device)

# early stop
stand = args.patience4earlystop
no_improvement_count = 0


# val_data
encoded_val = pd.DataFrame()
val_data = pd.read_csv(args.ValPath)
dataset_val = MingDataset(val_data, aa2idx)
dataloader_val = DataLoader(dataset_val, batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)

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
    print('编码采样后训练文件：', filename)
    df_train = pd.read_csv(os.path.join(data_storeage_folder, filename))
    encoded_file_train_all = pd.concat([encoded_file_train_all, df_train], ignore_index=True)

dataset_train = MingDataset(encoded_file_train_all, aa2idx)
dataloader_train = DataLoader(dataset_train, batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)

# 训练
model_tcrep = FNNemb()
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
    for cat_train, label_train in dataloader_train:
        # 将 cdr3_train 转换为张量

        label = torch.Tensor(label_train)
        label = label.to(device)

        seq, seq_lengths = pad_packed_sequence(cat_train, batch_first=True)
        seq = seq.squeeze().to(device)

        onehot_label = torch.eye(2).to(device)[label.long(), :].squeeze()
        # 独热编码且删除长度为1的维度
        output = model_tcrep(seq)
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
        for cat_val, label_val in dataloader_val:
            label = torch.Tensor(label_val)
            label = label.to(device)

            seq, seq_lengths = pad_packed_sequence(cat_val, batch_first=True)
            seq = seq.squeeze().to(device)

            out_put = model_tcrep(seq)
            onehot_label = torch.eye(2).to(device)[label.long(), :].squeeze()
            los_val = loss(out_put, onehot_label)
            loss_epoch_val.append(los_val.item())

            # print(loss_test)
            # 阈值设为0.5
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
        torch.save(best_model, 'model_save/FNNemb.pth')
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
