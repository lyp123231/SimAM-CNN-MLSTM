# -*- coding: utf-8 -*-
# 加载第三方库
import csv
import os
import time

# from torch.nn import ReLU
import numpy
from torch.utils.tensorboard import SummaryWriter

from test6 import target_super
import pandas as pd
import numpy as np
# 读取文件
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from MLSTM import MogLSTM as MLSTM
df1 = pd.read_csv("./NewData/{}.csv".format(1))
df2 = pd.read_csv("./NewData/{}.csv".format(2))
# df2 = pd.read_csv("./Two_Data/NewTwo/{}.csv".format(10))
# 合并
df = pd.concat([df1, df2])
# df.drop_duplicates()  #数据去重
# #保存合并后的文件
# df.to_csv('文件.csv',encoding = 'utf-8')
# if os.path.exists('./NewData/com.csv') == False:
df.to_csv("./NewData/com.csv", index=0, header=1)
rea = []
f = open("./NewData/com.csv", encoding="utf-8")
reader = csv.reader(f)
for k, i in enumerate(reader):
    if k > 0:
        rea.append(i)
data_gather = []
data_com = []
tar = []
num2 = 0

print("++++++++++++++++++++++++++++++++++++++")
for k, i in enumerate(rea):
    num2 += 1
    if int(num2) % target_super != 0:
        data_gather.append(i[1:-1])
    else:
        data_gather.append(i[1:-1])
        tar.append(i[-1])
        data_gather.append(tar)
        tar = []
        data_com.append(data_gather)
        data_gather = []
print(num2, "单笔画个数")
np.random.shuffle(data_com)
target = []
target = np.array(target)
temple = []
# print(type(data_com),type(data_com[0][0]))
for k, i in enumerate(data_com):
    temple.append(np.array(i[:-1]))
    target = np.append(target, i[-1])
    # break
print(np.unique(target))
print(pd.DataFrame(target).value_counts())
temple = np.array(temple, dtype=object)
# print(temple)
from sklearn.preprocessing import LabelEncoder  # 标签专用只允许一维数据

encorder = LabelEncoder().fit(target)
target = pd.DataFrame(encorder.transform(target)).values.ravel()
print(np.unique(target))
print(pd.DataFrame(target).value_counts())

# 下采样
# from imblearn.under_sampling import RandomUnderSampler
# cc = RandomUnderSampler(sampling_strategy={0:347,1:347},random_state=0)
# temple, target = cc.fit_resample(temple, target)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(temple, target, test_size=0.25, random_state=0)
# import keras as ks
train_data_size = len(Xtrain)
test_data_size = len(Xtest)
lenght = []
for i in temple:
    lenght.append(len(i))
# print(lenght)
train_data = Xtrain.astype(float)
train_data = torch.tensor(train_data)

test_data = Xtest.astype(float)
test_data = torch.tensor(test_data)

train_targets = Ytrain.astype(float)
train_targets = torch.tensor(train_targets)

test_targets = Ytest.astype(float)
test_targets = torch.tensor(test_targets)
train_set = TensorDataset(train_data, train_targets)
test_set = TensorDataset(test_data, test_targets)

BATCH_SIZE = 25
learning_rate = 0.0001
DataLoader_train_data = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
# print(DataLoader_train_data,"jhvfgyuk")
DataLoader_test_data = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[1, 2], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[1, 2], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class lstm(nn.Module):
    def __init__(self, input_size=40, hidden_size=1024, out_size=2, num_layer=2):
        super(lstm, self).__init__()

        self.BN = nn.BatchNorm1d(target_super)
        self.BN1 = nn.BatchNorm1d(2)
        #self.BN2 = nn.BatchNorm1d(1)
        self.att = simam_module()
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 8, 3),
            # nn.InstanceNorm1d(8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3),
            # nn.InstanceNorm1d(16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3),
            # nn.InstanceNorm1d(32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3),
            nn.InstanceNorm1d(64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.layer1 = MLSTM(input_size, 256, 4)
        self.layer4 = MLSTM(256, 64, 4)
        self.layer5 = nn.Linear(64, 2)
        self.fc5 = nn.Linear(576, 40)
        # self.layer3 = nn.MaxPool1d(1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, superCnn=None):
        if superCnn is None:
            superCnn = []
        for i in range(len(x[0])):
            y = x[:, i, :]
            y = y.view(len(y), 1, 48)
            y = self.BN1(y)
            y = self.att(y)
            y = self.layer3(y)
            y = self.fc5(y.view(y.size(0), -1))
            y = list(y.cpu().detach().numpy())
            superCnn.append(y)
        x = superCnn
        x = numpy.swapaxes(x, 0, 1)
        x = torch.tensor(x).to('cuda:0')
        x, (hn, hc) = self.layer1(x)
        x, (hn, hc) = self.layer4(x)
        hn = hn.view(1, len(hn), 64)
        ReLU = nn.ReLU()
        x = ReLU(hn[-1, :, :])
        x = self.layer5(x).squeeze(0)
        softmax = nn.Softmax(dim=0)
        x = softmax(x)
        return x

zh = lstm()
if torch.cuda.is_available():
    zh = zh.cuda()

# 定义优化器
# optimizer = torch.optim.Adam(zh.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(zh.parameters(), lr=learning_rate, weight_decay=0.001)
# ,weight_decay=0.001
# 设置训练网络的一些参数
# 设置训练的次数
total_train_step = 0
# 设置测试的次数
total_test_step = 0
# 设置训练的轮数
epoch = 300
# 添加tensorboard
writer = SummaryWriter("logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
max1 = []
max2 = []
max3 = []
for i in range(epoch):
    total_train_loss = 0
    total_train_acc = 0
    print("-------第{}轮训练开始-------".format(i + 1))
    # 训练步骤开始
    zh.train()
    for data in DataLoader_train_data:
        imgs, targets = data
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = zh(imgs)
        outputs = outputs.reshape((len(outputs), -1))
        loss = criterion(outputs, targets.long())
        total_train_loss = total_train_loss + loss
        # 优化器优化模型
        accuracy = (outputs.argmax(1) == targets).sum()  # 详情见tips_1.py
        total_train_acc = total_train_acc + accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        writer.add_scalar("train_loss", loss.item(), total_train_step)
    print("训练集的Loss:{}".format(total_train_loss))
    print("训练集的正确率：{}".format(total_train_acc / train_data_size))
    max1.append("{}".format(total_train_acc / train_data_size))
    zh.eval()
    # 测试步骤开始
    total_test_loss = 0
    total_acc = 0
    best_acc = 0
    best_acc_epo = 0
    with torch.no_grad():
        for data in DataLoader_test_data:
            imgs, targets = data
            imgs = imgs.to(torch.float32)
            targets = targets.to(torch.float32)
            # print(targets)
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zh(imgs)
            outputs = outputs.reshape((len(outputs), -1))
            loss = criterion(outputs, targets.long())
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()  # 详情见tips_1.py
            total_acc = total_acc + accuracy
            test_acc = total_acc / test_data_size
    print("测试集的Loss:{}".format(total_test_loss))
    print("测试集的正确率:{}".format(test_acc))
    max2.append("{}".format(test_acc))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_acc", total_acc / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
print("最大正确率训练集：" + max(max1))
print("最大正确率测试集：" + max(max2))
writer.close()
