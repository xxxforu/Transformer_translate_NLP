
# 使用可学习的词向量表（input & output embeddings）
# 将输入、输出索引映射为$d_{model}$-维词向量。词向量表可随机初始化，也可加载预训练词向量，如word2vec、glove等。

from torch import nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)