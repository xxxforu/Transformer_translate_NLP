import copy
import torch.nn as nn
from model.MultiHeadedAttention import MultiHeadedAttention
from model.Feed_Forward import PositionwiseFeedForward
from model.PositionalEncoding import PositionalEncoding
from model.EncoderLayer import EncoderLayer
from model.DecoderLayer import DecoderLayer
from model.Generator import Generator
from model.embedding import Embeddings
from model.Encoder import Encoder
from model.Decoder import Decoder
from model import config


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(config.DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(config.DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(config.DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(config.DEVICE), N).to(config.DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(config.DEVICE), N).to(config.DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(config.DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(config.DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(config.DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(config.DEVICE)