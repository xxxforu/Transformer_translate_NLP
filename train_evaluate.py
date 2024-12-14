import time
import torch
from model import config
from model.loss import SimpleLossCompute


def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    total_correct_tokens = 0  # 记录预测正确的Token数量
    tokens = 0.

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

        if i % 50 == 1:
            elapsed = time.time() - start
            accuracy = total_correct_tokens / total_tokens  # 计算当前Accuracy
            print("Epoch %d Batch: %d Loss: %f Accuracy: %.4f Tokens per Sec: %fs" % (
                epoch, i - 1, loss / batch.ntokens, accuracy, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    overall_accuracy = total_correct_tokens / total_tokens  # 整个epoch的Accuracy
    print("Epoch %d Overall Accuracy: %.4f" % (epoch, overall_accuracy))

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    """
    训练并保存模型
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5

    for epoch in range(config.EPOCHS):
        # 模型训练
        model.train()
        print(">>>>> Training")
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)

        # 模型评估
        model.eval()
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), config.SAVE_FILE)
            best_dev_loss = dev_loss
            print('****** Save model done... ******')

        print()
