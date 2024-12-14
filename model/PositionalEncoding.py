# 由于训练Transformer模型时，序列输入采用并行方式，因此缺少单词的位置信息，
# 通过在Transformer的输入中加入单词位置编码信息，使Transformer能够识别语句中单词的位置关系。
# 语句中每一个单词位置沿词向量维度由周期不同的正、余弦函数交替取值组合，生成独一纹理信息，从而使模型学到位置间的依赖关系和自然语言的时序特性。

import math
import torch
from torch import nn
from model import config
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device=config.DEVICE)
        # 单词位置
        position = torch.arange(0.0, max_len, device=config.DEVICE)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=config.DEVICE) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0 : : 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1 : : 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此设置requires_grad=False
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)