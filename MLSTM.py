from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class MogLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, mog_iteration):
        super(MogLSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iteration

        # 这里hiddensz乘4，是将四个门的张量运算都合并到一个矩阵当中，后续再通过张量分块给每个门
        self.Wih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))

        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()

    def init_weights(self):
        """
        权重初始化，对于W,Q,R使用xavier
        对于偏置b则使用0初始化
        :return:
        """
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt, ht):
        """
        计算mogrify
        :param xt:
        :param ht:
        :return:
        """
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R) * ht)
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q) * xt)
        return xt, ht

    def forward(self, x: torch.Tensor, init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:

            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states

        for t in range(seq_sz):
            xt = x[:, t, :]
            #print("uuuuuuuuuuu",t)
            xt, ht = self.mogrify(xt, ht)
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # chunk方法将tensor分块

            # LSTM
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            # outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            hidden_seq.append(ht.unsqueeze(Dim.batch))  # unsqueeze是给指定位置加上维数为1的维度
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (ht, Ct)
