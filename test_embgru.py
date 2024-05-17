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
from model import GRU32, CNNemb, GRUemb
from utils import get_dirlist, draw_roc, draw_plot
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

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




device = torch.device('cuda')


batch_size = 128
data_folder = r'C:\Users\86875\Desktop\learning\t-pred\data'
data_storeage_folder = r'C:\Users\86875\Desktop\learning\t-pred\code\data'

test_filenamelist = get_dirlist(data_folder, 'test')
test_filenamelist_sampled = get_dirlist(data_storeage_folder, 'test', 'sampled')

loss = nn.BCELoss()
loss = loss.to(device)

print('开始测试')

label_all = []
pred_all = []
output_all = []

encoded_file_test_all = pd.DataFrame()
for file_name in test_filenamelist_sampled:
    print('编码采样后训练文件：', file_name)
    df_train = pd.read_csv(os.path.join(data_storeage_folder, file_name))
    encoded_file_test_all = pd.concat([encoded_file_test_all, df_train], ignore_index=True)

dataset_test = MingDataset(encoded_file_test_all, aa2idx)
dataloader_test = DataLoader(dataset_test, batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)

model_path = r'C:\Users\86875\Desktop\learning\t-pred\code\ming\model better\gruemb\GRUemb.pth'
best_model = torch.load(model_path)
best_model = best_model.to(device)
best_model.eval()

loss_test_list = []

with torch.no_grad():
    for cat_test, label_test in dataloader_test:
        label = torch.Tensor(label_test)
        label = label.to(device)

        seq, seq_lengths = pad_packed_sequence(cat_test, batch_first=True)
        seq = seq.squeeze().to(device)

        onehot_label = torch.eye(2).to(device)[label.long(), :].squeeze()
        # 独热编码且删除长度为1的维度
        output = best_model(seq)
        los_test = loss(output, onehot_label)

        loss_test_list.append(los_test.item())
        pred_label = torch.argmax(output, dim=1)

        # acc需要numpy，先使用detach拷贝数据，再转到cpu上
        pred_cpu_numpy = pred_label.cpu().detach().numpy()
        label_cpu_numpy = label.cpu().detach().numpy()
        output_cpu_numpy = output[:, 1].cpu().detach().numpy()

        label_all.append(label_cpu_numpy)
        pred_all.append(pred_cpu_numpy)
        output_all.append(output_cpu_numpy)



loss = np.mean(loss_test_list)
label_test_all = np.concatenate(label_all)
pred_test_all = np.concatenate(pred_all)
output_test_all = np.concatenate(output_all)

fpr, tpr, _ = roc_curve(label_test_all, output_test_all)
roc_auc = auc(fpr, tpr)
pr, re, _ = precision_recall_curve(label_test_all, pred_test_all)
pr_re_auc = average_precision_score(label_test_all, pred_test_all)
f1 = f1_score(label_test_all, pred_test_all)
acc = accuracy_score(label_test_all, pred_test_all)
precision = precision_score(label_test_all, pred_test_all)
recall = recall_score(label_test_all, pred_test_all)


draw_roc(fpr, tpr, roc_auc, 'test_roc.png')
print(f'测试集F1：{f1}')
print(f'测试集ACC：{acc}')
print(f'测试集Precision：{precision}')
print(f'测试集Recall：{recall}')
print(f'测试集loss：{loss}')
