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
    data = PrepareData(config.TRAIN_FILE, config.DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    model = make_model(
        src_vocab,
        tgt_vocab,
        config.LAYERS,
        config.D_MODEL,
        config.D_FF,
        config.H_NUM,
        config.DROPOUT
    )
    model.load_state_dict(torch.load(config.SAVE_FILE))
    # 开始预测
    print(">>>>>>> start predict")
    evaluate_start = time.time()
    predict(data, model)
    print(f"<<<<<<< finished evaluate, cost {time.time() - evaluate_start:.4f} seconds")

if __name__ == '__main__':
    main()
