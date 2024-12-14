
import time
import torch
from model import config
from model.data_process import PrepareData
from model.Transformer import make_model
from model.LabelSmoothing import LabelSmoothing
from model.opt import NoamOpt
from train_evaluate import train
from predict import predict


def main():
    # 数据预处理
    data = PrepareData(config.TRAIN_FILE, config.DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    # 初始化模型
    model = make_model(
        src_vocab,
        tgt_vocab,
        config.LAYERS,
        config.D_MODEL,
        config.D_FF,
        config.H_NUM,
        config.DROPOUT
    )

    # 训练
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(config.D_MODEL, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    train(data, model, criterion, optimizer)
    print(f"<<<<<<< finished train, cost {time.time() - train_start:.4f} seconds")

    # 预测
    # 加载模型
    model.load_state_dict(torch.load(config.SAVE_FILE))
    # 开始预测
    print(">>>>>>> start predict")
    evaluate_start = time.time()
    predict(data, model)
    print(f"<<<<<<< finished evaluate, cost {time.time() - evaluate_start:.4f} seconds")


if __name__ == '__main__':
    main()
