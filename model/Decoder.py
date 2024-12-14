
# 解码器同样由$N$层解码器基本单元堆叠而成，与编码器基本单元不同的是：在编码器基本单元的多头**自注意力**机制及前馈网络之间插入一个**上下文注意力（context-attention）**机制（Multi-Head Attention）层，用解码器基本单元的自注意力机制输出作为$q$查询编码器的输出，以便解码时，解码器获得编码器的所有输出，即上下文注意力机制的$K$、$V$来自编码器的输出，$Q$来自解码器前一时刻的输出。
#
# 解码器基本单元的输入、输出：
#
# * 输入：编码器的输出、解码器前一时刻的输出
#
# * 输出：对应当前时刻输出单词的概率分布
#
# 此外，解码器的输出（最后一个解码器基本单元的输出）需要经线性变换和softmax函数映射为下一时刻预测单词的概率分布。
#
# 解码器**解码过程**：给定编码器输出（编码器输入语句所有单词的词向量）和解码器前一时刻输出（单词），预测当前时刻单词的概率分布。
#
# 注意：训练过程中，编、解码器均可以并行计算（训练语料中已知前一时刻单词）；推理过程中，编码器可以并行计算，解码器需要像RNN一样依次预测输出单词。


import torch.nn as nn
from model.MultiHeadedAttention import clones
from model.LayerNorm import LayerNorm

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        循环解码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)