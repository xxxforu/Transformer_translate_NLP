# 编码器注意力掩码的目地：使批次中较短语句的填充部分不参与注意力计算。
# 模型训练通常按批次进行，同一批次中的语句长度可能不同，因此需要按语句最大长度对短语句进行0填充以补齐长度。
# 语句填充部分属于无效信息，不应参与前向传播
#
# 解码器注意力掩码相对于编码器略微复杂，不仅需要将填充部分屏蔽掉，还需要对当前及后续序列进行屏蔽（subsequent_mask），
# 即解码器在预测当前时刻单词时，不能知道当前及后续单词内容，因此注意力掩码需要将当前时刻之后的注意力分数全部置为 −∞ ，
# 然后再计算 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 ，防止发生数据泄露。
# subsequent_mask的矩阵形式为一个下三角矩阵，在主对角线右上位置全部为False


import torch
import numpy as np
from model import config
from torch.autograd import Variable

def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """
    批次类
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """

    def __init__(self, src, trg=None, pad=config.PAD):
        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(config.DEVICE).long()
        trg = torch.from_numpy(trg).to(config.DEVICE).long()
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1]
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:]
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask