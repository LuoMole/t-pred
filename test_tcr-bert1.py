import os
import pickle as pk
import time
import numpy as np
import pandas as pd
import dt_process as dt
import torch

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score, \
    precision_recall_curve, average_precision_score, matthews_corrcoef
from torch import nn
from torch.optim import Adam, SGD, Adagrad, Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from utils import get_dirlist, draw_roc, draw_plot
from argparse import ArgumentParser
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_storeage_folder = r'C:\Users\86875\Desktop\learning\t-pred\code\ming\data-sampled'

test_filenamelist_sampled = get_dirlist(data_storeage_folder, 'data', 'comparison')

class MingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        m_cdr3 = str(self.df.iloc[idx]['CDR3']).upper()
        m_ep = str(self.df.iloc[idx]['Epitope']).upper()
        m_label = self.df.iloc[idx]['Label']
        cdr3_spaced = ' '.join(list(m_cdr3))
        ep_spaced = ' '.join(list(m_ep))
        return cdr3_spaced, ep_spaced, m_label


device = torch.device('cuda')
loss = nn.BCELoss()
loss = loss.to(device)

file_test_all = pd.DataFrame()
for filename in test_filenamelist_sampled:
    print('编码采样后训练文件：', filename)
    df_train = pd.read_csv(os.path.join(data_storeage_folder, filename))
    file_test_all = pd.concat([file_test_all, df_train], ignore_index=True)

dataset_test = MingDataset(file_test_all)
dataloader_test = DataLoader(dataset_test, 128, shuffle=True, drop_last=True)


tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
bertmodel = AutoModel.from_pretrained("wukevin/tcr-bert")
bertmodel = bertmodel.to(device)
#
class Bertcls(nn.Module):
    def __init__(self,cat_size=768, dropout=0):
        super().__init__()
        # self.cls = nn.Sequential(
        #     nn.Linear(768*2, 2),
        #     nn.Softmax(dim=-1))
        self.cls = nn.Sequential(
            nn.Linear(cat_size, cat_size//2),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(cat_size // 2, cat_size // 4),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(cat_size // 4, cat_size // 16),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(cat_size // 16, 2),
            nn.Softmax(dim=1)
        )


    def forward(self, cdr, ep):
        cdr = bertmodel(**cdr)
        cdr = cdr.last_hidden_state[:, 0, :]
        ep = bertmodel(**ep)
        ep = ep.last_hidden_state[:, 0, :]
        cat = torch.cat((cdr, ep), dim=-1)
        x = self.cls(cat)
        return x



# loaded_model = Bertcls()
# loaded = torch.load(r'C:\Users\86875\Desktop\learning\t-pred\code\ming\model better\bert\cls4tcr_bert.pth')
# loaded_model.load_state_dict(loaded.state_dict())
bert_model = torch.load(r'C:\Users\86875\Desktop\learning\t-pred\code\ming\model better\bert\cls4tcr_bert.pth').to(device)


label_all = []
pred_all = []
output_all = []
loss_test_list = []
with torch.no_grad():
    for cdr3_test, ep_test, label_test in dataloader_test:
        token_cdr = tokenizer(cdr3_test, return_tensors='pt', padding=True)
        token_cdr = token_cdr.to(device)
        token_ep = tokenizer(ep_test, return_tensors='pt', padding=True)
        token_ep = token_ep.to(device)
        # print(token_pair)
        out_put = bert_model(token_cdr, token_ep)


        label = torch.Tensor(label_test)
        label = label.to(device)
        onehot_label = torch.eye(2).to(device)[label.long(), :].squeeze()

        los_test = loss(out_put, onehot_label)
        loss_test_list.append(los_test.item())

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
loss = np.mean(loss_test_list)
label_test_all = np.concatenate(label_all)
pred_test_all = np.concatenate(pred_all)
output_test_all = np.concatenate(output_all)
mcc = matthews_corrcoef(y_true=label_test_all, y_pred=pred_test_all)

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
print(f'测试集mcc：{mcc}')
print(f'测试集auc：{roc_auc}')