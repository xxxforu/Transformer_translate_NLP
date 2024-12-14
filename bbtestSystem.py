import time
import torch
from model import config
from model.data_process import PrepareData
from model.Transformer import make_model
from model.LabelSmoothing import LabelSmoothing
from model.opt import NoamOpt
from train_evaluate import train
from predict import greedy_decode
from tkinter import *
from tkinter import messagebox
from torch.autograd import Variable
from model.Attention_mask import subsequent_mask


def predict_single_sentence(data, model, input_text):
    """
    对单个英文句子进行预测翻译
    """
    # 将输入的英文句子分词，并转换为对应的ID
    input_tokens = input_text.split()
    input_ids = [data.en_word_dict.get(token, data.en_word_dict["UNK"]) for token in input_tokens]

    # 将输入转换为tensor并移到指定设备（GPU/CPU）
    src = torch.tensor(input_ids).long().unsqueeze(0).to(config.DEVICE)

    # 创建输入的mask
    src_mask = (src != 0).unsqueeze(-2)

    # 使用训练好的模型进行翻译
    out = greedy_decode(model, src, src_mask, max_len=config.MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])

    # 处理模型输出的翻译结果
    translation = []
    for j in range(1, out.size(1)):  # 从1开始，跳过BOS标记
        sym = data.cn_index_dict[out[0, j].item()]
        if sym != 'EOS':  # 如果输出字符不为EOS终止符，则加入翻译结果
            translation.append(sym)
        else:
            break

    return " ".join(translation)


def run_prediction(data, model, input_text, result_var):
    """
    处理用户输入并运行翻译任务
    """
    # 获取用户输入的英文句子
    user_input = input_text.get()

    if user_input.strip() == "":
        messagebox.showerror("输入错误", "请输入一个句子进行翻译！")
        return

    # 使用训练好的模型进行翻译
    try:
        # model.load_state_dict(torch.load(config.SAVE_FILE))
        # 在pp电脑环境下会报错，改为下面：
        model.load_state_dict(torch.load(config.SAVE_FILE, map_location=torch.device('cpu')))
        print("Model loaded successfully!")
    except Exception as e:
        messagebox.showerror("加载模型失败", f"模型加载失败: {e}")
        return

    # 执行翻译
    translation = predict_single_sentence(data, model, user_input)

    # 显示翻译结果
    result_var.set(f"翻译结果: {translation}")


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

    # 创建一个图形界面
    root = Tk()
    root.title("基于Transformer的机器翻译")

    # 设置窗口大小
    root.geometry("600x300")

    # 输入框标签
    input_label = Label(root, text="请输入要翻译的英文句子:")
    input_label.pack(pady=10)

    # 创建一个输入框
    input_text = Entry(root, width=50)
    input_text.pack(pady=10)

    # 创建一个显示翻译结果的标签
    result_var = StringVar()
    result_label = Label(root, textvariable=result_var, width=50, height=3, anchor="w", bg="lightgray")
    result_label.pack(pady=10)

    # 创建一个按钮，点击后进行翻译
    translate_button = Button(root, text="开始翻译", command=lambda: run_prediction(data, model, input_text, result_var))
    translate_button.pack(pady=10)

    # 启动界面
    root.mainloop()


if __name__ == '__main__':
    main()
