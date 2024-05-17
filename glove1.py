import math
import os
import sys
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import get_dirlist

#加载语料库


#计算共现矩阵
def buildCooccuranceMatrix(text, word_to_idx):
    vocab_size = len(word_to_idx)
    maxlength = len(text)
    text_ids = [word_to_idx.get(word) for word in text]
    cooccurance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    # print("Co-Matrix consumed mem:%.2fMB" % (sys.getsizeof(cooccurance_matrix)/(1024*1024)))
    for i, center_word_id in enumerate(text_ids):
        if center_word_id is None:
            a = text[i]
            print(a)
        window_indices = list(range(i - WINDOW_SIZE, i)) + list(range(i + 1, i + WINDOW_SIZE + 1))
        # 实际上是对window_indices中所有元素验证了一遍不会超过maxlength的长度
        window_indices = [i % maxlength for i in window_indices]
        window_word_ids = [text_ids[index] for index in window_indices]
        for context_word_id in window_word_ids:
            # print(context_word_id)
            # print(window_word_ids)
            cooccurance_matrix[center_word_id][context_word_id] += 1
        # if (i+1) % 1000000 == 0:
        #     print(">>>>> Process %dth word" % (i+1))
    # print(">>>>> Save co-occurance matrix completed.")
    return cooccurance_matrix


#计算权重函数矩阵
def buildWeightMatrix(co_matrix):
    #此处100改为500
    xmax = 10
    weight_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    # print("Weight-Matrix consumed mem:%.2fMB" % (sys.getsizeof(weight_matrix) / (1024 * 1024)))
    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            weight_matrix[i][j] = math.pow(co_matrix[i][j] / xmax, 0.75) if co_matrix[i][j] < xmax else 1
        # if (i+1) % 1000 == 0:
        #     print(">>>>> Process %dth weight" % (i+1))
    # print(">>>>> Save weight matrix completed.")
    return weight_matrix

#创建dataloader
class WordEmbeddingDataset(Dataset):
    def __init__(self, co_matrix, weight_matrix):
        self.co_matrix = co_matrix
        self.weight_matrix = weight_matrix
        self.train_set = []

        for i in range(self.weight_matrix.shape[0]):
            for j in range(self.weight_matrix.shape[1]):
                if weight_matrix[i][j] != 0:
                    # 这里对权重进行了筛选，去掉权重为0的项
                    # 因为共现次数为0会导致log(X)变成nan
                    self.train_set.append((i, j))

    def __len__(self):
        '''
        必须重写的方法
        :return: 返回训练集的大小
        '''
        return len(self.train_set)

    def __getitem__(self, index):
        '''
        必须重写的方法
        :param index:样本索引
        :return: 返回一个样本
        '''
        (i, j) = self.train_set[index]
        return i, j, torch.tensor(self.co_matrix[i][j], dtype=torch.float), self.weight_matrix[i][j]

#创建训练模型
class GloveModelForBGD(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # 声明v和w为Embedding向量
        self.v = nn.Embedding(vocab_size, embed_size)
        self.w = nn.Embedding(vocab_size, embed_size)
        self.biasv = nn.Embedding(vocab_size, 1)
        self.biasw = nn.Embedding(vocab_size, 1)

        # 随机初始化参数
        initrange = 0.5 / self.embed_size
        self.v.weight.data.uniform_(-initrange, initrange)
        self.w.weight.data.uniform_(-initrange, initrange)

    def forward(self, i, j, co_occur, weight):
        # 根据目标函数计算Loss值
        vi = self.v(i)  # 分别根据索引i和j取出对应的词向量和偏差值
        wj = self.w(j)
        bi = self.biasv(i)
        bj = self.biasw(j)

        similarity = torch.mul(vi, wj)
        similarity = torch.sum(similarity, dim=1)

        loss = similarity + bi + bj - torch.log(co_occur)
        loss = 0.5 * weight * loss * loss

        return loss.sum().mean()

    def gloveMatrix(self):
        '''
        获得词向量，这里把两个向量相加作为最后的词向量
        :return:
        '''
        return self.v.weight.data.numpy() + self.w.weight.data.numpy()

#训练
class ming(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        m_cdr3 = str(self.df.iloc[idx]['CDR3']).upper()
        return m_cdr3


EMBEDDING_SIZE = 32		#50个特征
MAX_VOCAB_SIZE = 23	#词汇表大小为2000个词语
WINDOW_SIZE = 2			#窗口大小为5

NUM_EPOCHS = 10		#迭代10次
BATCH_SIZE = 10			#一批有10个样本
LEARNING_RATE = 0.0001	#初始学习率
TEXT_SIZE = 20000000	#控制从语料库读取语料的规模

# aas = 'ACDEFGHIKLMNPQRSTVWY'
aas = '*ACDEFGHIKLMNOPQRSTVWXY'
aa2idx = {aas[i]: i for i in range(len(aas))}
idx2aa = {i: aa for aa, i in aa2idx.items()}

# data_storeage_folder = r'C:\Users\86875\Desktop\learning\t-pred\code\data'
data_storeage_folder = 'data-sampled'
# train_filenamelist_sampled = get_dirlist(data_storeage_folder, 'train', 'sampled')
train_filenamelist_sampled = get_dirlist(data_storeage_folder, 'train', '14')
text_all = list()

model = GloveModelForBGD(MAX_VOCAB_SIZE, EMBEDDING_SIZE) #创建模型
optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE) #选择Adagrad优化器

epochs = NUM_EPOCHS
# 加载语料及预处理
for filename in train_filenamelist_sampled:
    print('编码采样后训练文件：', filename)
    df_train = pd.read_csv(os.path.join(data_storeage_folder, filename))
    # tcr = df_train['CDR3'].tolist()
    dataset1 = ming(df_train)
    dataloader1 = DataLoader(dataset1, 256, shuffle=True, drop_last=True)
    for tcr in tqdm(dataloader1):
        tcr = [letter.upper() for word in tcr for letter in word]
        co_matrix = buildCooccuranceMatrix(tcr, aa2idx)  # 构建共现矩阵
        weight_matrix = buildWeightMatrix(co_matrix)  # 构建权重矩阵
        dataset = WordEmbeddingDataset(co_matrix, weight_matrix)  # 创建dataset
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        for epoch in range(epochs):
            loss_print_avg = 0
            for i, j, co_occur, weight in dataloader:
                loss = model(i, j, co_occur, weight)  # 前向传播
                optimizer.zero_grad()  # 每一批样本训练前重置缓存的梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新梯度
                loss_print_avg += loss.item()





torch.save(model, 'model_save/glove_comparison.pth')

aa_embeddings = model.gloveMatrix()
ordered_aa_embeddings = np.array([aa_embeddings[aa2idx[aa]] for aa in aas])
zero_embedding = np.zeros((32,))
embedding_dict = dict(zip(aas, ordered_aa_embeddings))
embedding_dict[' '] = zero_embedding
with open('emb/aa_glove_comparison.pk', 'wb') as file:
    pickle.dump(embedding_dict, file)
