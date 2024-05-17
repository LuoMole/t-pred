import torch
from torch import nn
from itertools import repeat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


#备选
# class tcrnet(nn.Module):
#     def __init__(self,device="cuda:0"):
#         super().__init__()
#         self.device = device
#         self.project = nn.Sequential(
#             nn.Conv1d(6, 128, 3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, 64, 3),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Conv1d(64, 32, 3),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(38*32, 560),
#             nn.Linear(560, 280),
#             nn.Linear(280, 96),
#             nn.Linear(96, 1),
#             nn.LayerNorm(1)
#
#         )
#     def forward(self, x):
#         x = self.project(x)
#         x = torch.sigmoid(x)
#         return x
#
#
# #input_shape(15(ep_len),6(因子数）)
# class epnet(nn.Module):
#     def __init__(self,device="cuda:0"):
#         super().__init__()
#         self.device = device
#         self.project = nn.Sequential(
#             nn.Conv1d(6, 128, 3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, 64, 3),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(11*64, 96),
#             nn.Linear(96, 48),
#             nn.Linear(48, 1),
#             nn.LayerNorm(1)
#
#         )
#     def forward(self, x):
#         x = self.project(x)
#         x = torch.sigmoid(x)
#         return x
# if __name__ == '__main__':#确保创建实例这个过程不会在导入该文件的时候执行
#     tcr_model = tcrnet()
#     ep_model = epnet()

# class TcrEpNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # inputsize tcr(batchsize, 20, 44) ep(batchsize, 20, 15)
#         self.sigmoid = nn.Sigmoid()
#
#         self.conv1 = nn.Conv1d(20, 18, 1, padding='same')
#         nn.init.xavier_uniform_(self.conv1.weight)
#         # self.bn1 = nn.BatchNorm1d(18, device='cuda')
#         # self.convA = nn.Conv1d(18, 36, 1, padding='same')
#         # nn.init.xavier_uniform_(self.convA.weight)
#         # self.bnA = nn.BatchNorm1d(12)
#
#         self.conv2 = nn.Conv1d(20, 18, 3, padding='same')
#         nn.init.xavier_uniform_(self.conv2.weight)
#         #self.bn2 = nn.BatchNorm1d(16)
#         # self.convB = nn.Conv1d(18, 36, 3, padding='same')
#         # nn.init.xavier_uniform_(self.convB.weight)
#         # self.bnB = nn.BatchNorm1d(12)
#
#         self.conv3 = nn.Conv1d(20, 18, 5, padding='same')
#         nn.init.xavier_uniform_(self.conv3.weight)
#         #self.bn3 = nn.BatchNorm1d(16)
#         # self.convC = nn.Conv1d(18, 36, 5, padding='same')
#         # nn.init.xavier_uniform_(self.convC.weight)
#         # self.bnC = nn.BatchNorm1d(12)
#
#         self.conv4 = nn.Conv1d(20, 18, 7, padding='same')
#         nn.init.xavier_uniform_(self.conv4.weight)
#         #self.bn4 = nn.BatchNorm1d(16)
#         # self.convD = nn.Conv1d(18, 36, 7, padding='same')
#         # nn.init.xavier_uniform_(self.convD.weight)
#         # self.bnD = nn.BatchNorm1d(12)
#
#         self.conv5 = nn.Conv1d(20, 18, 9, padding='same')
#         nn.init.xavier_uniform_(self.conv5.weight)
#         #self.bn5 = nn.BatchNorm1d(16)
#         # self.convE = nn.Conv1d(18, 36, 9, padding='same')
#         # nn.init.xavier_uniform_(self.convE.weight)
#         # self.bnE = nn.BatchNorm1d(12)
#
#         # self.conv6 = nn.Conv1d(20, 18, 11, padding='same')
#         # nn.init.xavier_uniform_(self.conv6.weight)
#         # self.bn6 = nn.BatchNorm1d(16)
#         # self.convF = nn.Conv1d(6, 12, 13, padding='same')
#         # nn.init.xavier_uniform_(self.convF.weight)
#         # self.bnF = nn.BatchNorm1d(12)
#
#         # self.conv7 = nn.Conv1d(6, 6, 13, padding='same')
#         # nn.init.xavier_uniform_(self.conv5.weight)
#         # self.bn7 = nn.BatchNorm1d(6)
#         self.pool = nn.AdaptiveMaxPool1d(1)# 全局最大池化
#         self.drop = nn.Dropout(0.3)
#         self.flat = nn.Flatten()
#         self.l1 = nn.Linear(10*18, 32)
#         self.l2 = nn.Linear(32, 1)
#
#
#     def forward(self, cdr, ep):
#         cdr_x1 = self.conv1(cdr)
#         #为了实现全局最大池化
#         # cdr_x1 = self.bn1(cdr_x1)
#         cdr_x1 = self.sigmoid(cdr_x1)
#         #cdr_x1 = self.convA(cdr_x1)
#         # cdr_x1 = self.bnA(cdr_x1)
#         #cdr_x1 = self.sigmoid(cdr_x1)
#         cdr_x1 = self.pool(cdr_x1)
#         # cdr_x1 = nn.functional.max_pool1d(cdr_x1, kernel_size=(cdr_x1.shape[2]))
#         #cdr_x1 = self.drop(cdr_x1)
#
#         cdr_x2 = self.conv2(cdr)
#         #cdr_x2 = self.bn2(cdr_x2)
#         cdr_x2 = self.sigmoid(cdr_x2)
#         #cdr_x2 = self.convB(cdr_x2)
#         # cdr_x2 = self.bnB(cdr_x2)
#         #cdr_x2 = self.sigmoid(cdr_x2)
#         cdr_x2 = self.pool(cdr_x2)
#         # cdr_x2 = nn.functional.max_pool1d(cdr_x2, kernel_size=(cdr_x2.shape[2]))
#         #cdr_x2 = self.drop(cdr_x2)
#
#         cdr_x3 = self.conv3(cdr)
#         #cdr_x3 = self.bn3(cdr_x3)
#         cdr_x3 = self.sigmoid(cdr_x3)
#         #cdr_x3 = self.convC(cdr_x3)
#         # cdr_x3 = self.bnC(cdr_x3)
#         #cdr_x3 = self.sigmoid(cdr_x3)
#         cdr_x3 = self.pool(cdr_x3)
#         # cdr_x3 = nn.functional.max_pool1d(cdr_x3, kernel_size=(cdr_x3.shape[2]))
#         #cdr_x3 = self.drop(cdr_x3)
#
#         cdr_x4 = self.conv4(cdr)
#         #cdr_x4 = self.bn4(cdr_x4)
#         cdr_x4 = self.sigmoid(cdr_x4)
#         #cdr_x4 = self.convD(cdr_x4)
#         # cdr_x4 = self.bnB(cdr_x4)
#         #cdr_x4 = self.sigmoid(cdr_x4)
#         cdr_x4 = self.pool(cdr_x4)
#         # cdr_x4 = nn.functional.max_pool1d(cdr_x4, kernel_size=(cdr_x4.shape[2]))
#         #cdr_x4 = self.drop(cdr_x4)
#
#         cdr_x5 = self.conv5(cdr)
#         #cdr_x5 = self.bn5(cdr_x5)
#         cdr_x5 = self.sigmoid(cdr_x5)
#         #cdr_x5 = self.convE(cdr_x5)
#         # cdr_x5 = self.bnB(cdr_x5)
#         #cdr_x5 = self.sigmoid(cdr_x5)
#         cdr_x5 = self.pool(cdr_x5)
#         # cdr_x5 = nn.functional.max_pool1d(cdr_x5, kernel_size=(cdr_x5.shape[2]))
#         #cdr_x5 = self.drop(cdr_x5)
#
#         # cdr_x6 = self.conv6(cdr)
#         #cdr_x6 = self.bn6(cdr_x6)
#         # cdr_x6 = self.sigmoid(cdr_x6)
#         # cdr_x6 = self.convF(cdr_x6)
#         # cdr_x6 = self.bnB(cdr_x6)
#         # cdr_x6 = self.sigmoid(cdr_x6)
#         # cdr_x6 = self.pool(cdr_x6)
#         #cdr_x6 = nn.functional.max_pool1d(cdr_x6, kernel_size=(cdr_x6.shape[2]))
#         #cdr_x6 = self.drop(cdr_x6)
#
#         # cdr_x7 = self.conv6(cdr)
#         # cdr_x7 = self.bn6(cdr_x7)
#         # cdr_x7 = torch.sigmoid(cdr_x7)
#         # cdr_x7 = nn.functional.max_pool1d(cdr_x7, kernel_size=(cdr_x7.shape[2]))
#
#         cdr_x = torch.cat([cdr_x1, cdr_x2, cdr_x3, cdr_x4, cdr_x5], dim=-1)
#
#         ep_x1 = self.conv1(ep)
#         ep_x1 = self.sigmoid(ep_x1)
#         #ep_x1 = self.convA(ep_x1)
#         # ep_x1 = self.bnA(ep_x1)
#         #ep_x1 = self.sigmoid(ep_x1)
#         ep_x1 = self.pool(ep_x1)
#         # ep_x1 = nn.functional.max_pool1d(ep_x1, kernel_size=(ep_x1.shape[2]))
#         #ep_x1 = self.drop(ep_x1)
#
#         ep_x2 = self.conv2(ep)
#         #ep_x2 = self.bn2(ep_x2)
#         ep_x2 = self.sigmoid(ep_x2)
#         #ep_x2 = self.convB(ep_x2)
#         # ep_x2 = self.bnB(ep_x2)
#         #ep_x2 = self.sigmoid(ep_x2)
#         ep_x2 = self.pool(ep_x2)
#         # ep_x2 = nn.functional.max_pool1d(ep_x2, kernel_size=(ep_x2.shape[2]))
#         #ep_x2 = self.drop(ep_x2)
#
#         ep_x3 = self.conv3(ep)
#         #ep_x3 = self.bn3(ep_x3)
#         ep_x3 = self.sigmoid(ep_x3)
#         #ep_x3 = self.convC(ep_x3)
#         # ep_x3 = self.bnC(ep_x3)
#         #ep_x3 = self.sigmoid(ep_x3)
#         ep_x3 = self.pool(ep_x3)
#         # ep_x3 = nn.functional.max_pool1d(ep_x3, kernel_size=(ep_x3.shape[2]))
#         #ep_x3 = self.drop(ep_x3)
#
#         ep_x4 = self.conv4(ep)
#         #ep_x4 = self.bn4(ep_x4)
#         ep_x4 = self.sigmoid(ep_x4)
#         #ep_x4 = self.convD(ep_x4)
#         # ep_x4 = self.bnD(ep_x4)
#         #ep_x4 = self.sigmoid(ep_x4)
#         ep_x4 = self.pool(ep_x4)
#         # ep_x4 = nn.functional.max_pool1d(ep_x4, kernel_size=(ep_x4.shape[2]))
#         #ep_x4 = self.drop(ep_x4)
#
#         ep_x5 = self.conv5(ep)
#         #ep_x5 = self.bn5(ep_x5)
#         ep_x5 = self.sigmoid(ep_x5)
#         #ep_x5 = self.convE(ep_x5)
#         # ep_x5 = self.bnD(ep_x5)
#         #ep_x5 = self.sigmoid(ep_x5)
#         ep_x5 = self.pool(ep_x5)
#         # ep_x5 = nn.functional.max_pool1d(ep_x5, kernel_size=(ep_x5.shape[2]))
#         #ep_x5 = self.drop(ep_x5)
#
#         # ep_x6 = self.conv5(ep)
#         # ep_x6 = self.bn6(ep_x6)
#         # ep_x6 = torch.sigmoid(ep_x6)
#         # ep_x6 = nn.functional.max_pool1d(ep_x6, kernel_size=(ep_x6.shape[2]))
#         # ep_x6 = self.drop(ep_x6)
#
#         ep_x = torch.cat([ep_x1, ep_x2, ep_x3, ep_x4, ep_x5], dim=-1)
#
#         cat = torch.cat([cdr_x, ep_x], dim=2)
#         cat = self.flat(cat)
#         l1 = self.l1(cat)
#         l1 = self.sigmoid(l1)
#         l1 = self.drop(l1)
#         l2 = self.l2(l1)
#         l2 = self.sigmoid(l2)
#         return l2


class GRU32(nn.Module):
    def __init__(self, cat_size=1536, embeddim=32, dropout=0):
        super().__init__()
        self.tcr_GRU = nn.GRU(embeddim, cat_size//4, 3, batch_first=True, bidirectional=True)
        # 只能用// 因为/会返回普通除法，造成浮点数结果（比如3.0）这是gru不能接受的
        self.ep_GRU = nn.GRU(embeddim, cat_size//4, 3, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
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

            # nn.Sigmoid()
            #softmax改为sigmoid
        )

    def forward(self, tcr, ep):
        tcr, _ = self.tcr_GRU(tcr)
        ep, _ = self.ep_GRU(ep)
        tcr = tcr[ : , -1, : ]
        ep = ep[:, -1, :]
        cat = torch.cat((tcr, ep), dim=-1)
        output = self.projection(cat)
        return output
# cr = GRU32()


# class Embcnn(nn.Module)
# class CNNemb(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Embedding(21, 32),
#             nn.Conv1d(32, 128, 3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(0.35),
#             nn.Flatten(),
#             nn.Linear(128, 1)
#         )
#     def forward(self, seq):
#         output = self.projection(seq)
#         return output
#
# import torch.nn as nn


#emb模型结尾不同
class CNNemb(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(21, 32)
        self.conv1d = nn.Conv1d(32, 128, 3, padding='same')
        self.batchnorm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.35)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

        self.l1 = nn.Linear(3712, 1856)
        self.l2 = nn.Linear(1856, 464)
        self.l3 = nn.Linear(464, 232)
        self.l4 = nn.Linear(232, 2)
        # 二值分类

    def forward(self, seq):
        x = self.embedding(seq)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.l3(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.l4(x)
        x = self.softmax(x)
        return x


class GRUemb(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(21, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.layernorm1 = nn.LayerNorm(128)
        self.dropout1d = nn.Dropout1d(0.4)
        self.dropout = nn.Dropout(0.35)

        self.GRu = nn.GRU(128, 128, num_layers=2, dropout=0.2, batch_first=True)#torch中无循环dropout
        self.batchnorm2 = nn.LayerNorm(256)
        self.l1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, 1024)
        self.relu = nn.ReLU()
        self.batchnorm3 = nn.LayerNorm(1024)
        self.l3 = nn.Linear(1024, 1024)
        self.l4 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, seq):
        x = self.embedding(seq)
        x = self.layernorm1(x)
        x = x.permute(0, 2, 1)
        x = self.dropout1d(x)
        x = x.permute(0, 2, 1)
        x, _ = self.GRu(x)
        x = x[ : , -1, : ]
        x = self.batchnorm1(x)
        x = self.l1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batchnorm2(x)
        x = self.l2(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.l3(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.l4(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.l5(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.l6(x)
        x = self.softmax(x)
        return x

class FNNemb(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(21, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.layernorm1 = nn.LayerNorm(128)
        self.dropout1d = nn.Dropout1d(0.4)
        self.dropout = nn.Dropout(0.35)

        self.batchnorm2 = nn.LayerNorm(256)
        self.l1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, 1024)
        self.relu = nn.ReLU()
        self.batchnorm3 = nn.LayerNorm(1024)
        self.l3 = nn.Linear(1024, 1024)
        self.l4 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 2)
        self.l7 = nn.Linear(118, 2)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()


    def forward(self, seq):
        x = self.embedding(seq)
        x = self.layernorm1(x)
        x = self.dropout1d(x)
        x = self.l1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batchnorm2(x)
        x = self.l2(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        x = self.batchnorm3(x)
        x = self.l4(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.l5(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.l6(x)
        x = self.flatten(x)
        x = self.l7(x)
        x = self.softmax(x)
        return x

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(21, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.layernorm1 = nn.LayerNorm(128)
        self.dropout1d = nn.Dropout1d(0.4)
        self.dropout = nn.Dropout(0.35)

        self.batchnorm2 = nn.LayerNorm(256)
        self.l1 = nn.Linear(21, 256)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, 1024)
        self.relu = nn.ReLU()
        self.batchnorm3 = nn.LayerNorm(1024)
        self.l3 = nn.Linear(1024, 1024)
        self.l4 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 2)
        self.l7 = nn.Linear(118, 2)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()


    def forward(self, seq):
        x = self.l1(seq)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batchnorm2(x)
        x = self.l2(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        x = self.batchnorm3(x)
        x = self.l4(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.l5(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.l6(x)
        x = self.flatten(x)
        x = self.l7(x)
        x = self.softmax(x)
        return x
cls_dropout = 0


class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Dropout(0.1, inplace=False),
            nn.Linear(1536, 2),
            # nn.Dropout(cls_dropout),
            # nn.ReLU(),
            # nn.Linear(768, 384),
            # nn.Dropout(cls_dropout),
            # nn.ReLU(),
            # nn.Linear(384, 128),
            # nn.Dropout(cls_dropout),
            # nn.ReLU(),
            # nn.Linear(128, 2),
            nn.Softmax(dim=-1))

    def forward(self, input):
        x = self.cls(input)
        return x
