# 训练过程中，采用KL散度损失实现标签平滑（$\epsilon_{ls} = 0.1$）策略，提高模型鲁棒性、准确性和BLEU分数。
# 标签平滑：输出概率分布由one-hot方式转为真实标签的概率置为`confidence`，其它所有非真实标签概率平分`1 - confidence`。

import torch.nn as nn
import torch
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    """
    标签平滑
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
