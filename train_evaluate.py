import time
import torch
from model import config
from model.loss import SimpleLossCompute
from torchtext.data.metrics import bleu_score

def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    total_correct_tokens = 0  # 记录预测正确的Token数量
    tokens = 0.

    all_refs = []  # 存储真实翻译
    all_hyp = []  # 存储模型翻译

    for i, batch in enumerate(data):
        # 模型前向传播
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        # 计算预测结果
        pred = torch.argmax(out, dim=-1)  # 获取模型预测的最大概率索引
        correct_tokens = (pred == batch.trg_y).sum().item()  # 统计正确的Token数量

        total_correct_tokens += correct_tokens  # 累积正确Token数量
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # 将批次的翻译结果按句子逐一保存
        refs = batch.trg_y.tolist()  # 真实翻译
        hyps = pred.tolist()  # 预测翻译

        all_refs.extend(refs)
        all_hyp.extend(hyps)

        if i % 50 == 1:
            elapsed = time.time() - start
            accuracy = total_correct_tokens / total_tokens  # 计算当前Accuracy
            print("Epoch %d Batch: %d Loss: %f Accuracy: %.4f Tokens per Sec: %fs" % (
                epoch, i - 1, loss / batch.ntokens, accuracy, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    # 计算BLEU得分
    bleu = bleu_score(all_hyp, all_refs)
    print("Epoch %d BLEU Score: %.4f" % (epoch, bleu))

    overall_accuracy = total_correct_tokens / total_tokens  # 整个epoch的Accuracy
    print("Epoch %d Overall Accuracy: %.4f" % (epoch, overall_accuracy))

    return total_loss / total_tokens


def train(data, model, criterion, optimizer, tokenizer=None):
    """
    训练并保存模型
    """
    best_dev_loss = 1e5

    for epoch in range(config.EPOCHS):
        model.train()
        print(">>>>> Training")
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)

        model.eval()
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), config.SAVE_FILE)
            best_dev_loss = dev_loss
            print('****** Save model done... ******')

        print()
