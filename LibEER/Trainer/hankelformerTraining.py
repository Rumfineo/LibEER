import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.metric import Metric
from utils.store import save_state

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, repr1, repr2):
        z1 = F.normalize(repr1, dim=1)
        z2 = F.normalize(repr2, dim=1)
        logits12 = (z1 @ z2.t()) / self.temperature
        logits21 = (z2 @ z1.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss12 = F.cross_entropy(logits12, labels)
        loss21 = F.cross_entropy(logits21, labels)
        return 0.5 * (loss12 + loss21)

def train(model,
          train_tuple, val_tuple, test_tuple, # 修改为接收 index_to_data 返回的元组
          device,
          output_dir="result/",
          metrics=None, metric_choose=None,
          optimizer=None, scheduler=None,
          batch_size=16, epochs=40,
          criterion=None,
          contrastive_criterion=None, lambda_contrastive=0.0,
          lambda_domain=0.1): # 新增：领域对抗损失权重

    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]

    # 1. 根据 split.py 的返回逻辑封装 TensorDataset
    # train_tuple: (data, label, subj, ...)
    def make_dataset(tup):
        return TensorDataset(
            torch.FloatTensor(tup[0]),
            torch.LongTensor(tup[1]),
            torch.LongTensor(tup[2])
        )

    dataset_train = make_dataset(train_tuple)
    dataset_val = make_dataset(val_tuple)
    dataset_test = make_dataset(test_tuple)

    data_loader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
    data_loader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

    model = model.to(device)
    best_metric = {s: 0. for s in metrics}

    for epoch in range(epochs):
        model.train()
        metric = Metric(metrics)

        train_bar = tqdm(
            enumerate(data_loader_train),
            total=len(data_loader_train),
            desc=f"Train Epoch {epoch}/{epochs}"
        )

        for idx, (samples, targets, domain_labels) in train_bar:
            samples, targets, domain_labels = samples.to(device), targets.to(device), domain_labels.to(device)

            # 1. 如果 targets 是 [Batch, Num_Classes] (One-hot)，转为索引 [Batch]
            if targets.dim() > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            # 2. 如果 targets 是 [Batch, 1]，去掉冗余维度变成 [Batch]
            elif targets.dim() > 1:
                targets = targets.view(-1)

            targets = targets.long() # 确保是 Long 类型

            # 领域标签同理
            if domain_labels.dim() > 1:
                domain_labels = domain_labels.view(-1)
            domain_labels = domain_labels.long()

            # 2. DANN Alpha 调度逻辑：随进度从 0 增加到 1
            p = float(idx + epoch * len(data_loader_train)) / (epochs * len(data_loader_train))
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 0.5

            optimizer.zero_grad()

            # 模型返回：(情感logits, 领域logits, 序列表征1, 序列表征2)
            out = model(samples, alpha=alpha)

            # 兼容性处理
            if len(out) == 4:
                logits, domain_logits, repr1, repr2 = out
            else:
                logits, repr1, repr2 = out
                domain_logits = None

            # 计算损失
            cls_loss = criterion(logits, targets)

            # 领域损失计算 (只对正值 ID 计算)
            dom_loss = torch.zeros((), device=device)
            if domain_logits is not None and lambda_domain > 0:
                dom_loss = F.cross_entropy(domain_logits, domain_labels)

            # 对比学习损失
            con_loss = torch.zeros((), device=device)
            if (contrastive_criterion is not None) and (lambda_contrastive > 0) and (repr1 is not None):
                con_loss = contrastive_criterion(repr1, repr2)

            total_loss = cls_loss + (lambda_contrastive * con_loss) + (lambda_domain * dom_loss)

            # 更新准确率统计 (参考正确代码使用 .eq)
            _, pred = torch.max(logits, dim=1)
            metric.update(pred, targets)
            train_bar.set_postfix_str(
                f"L:{total_loss.item():.4f} Cls:{cls_loss.item():.3f} Dom:{dom_loss.item():.3f}"
            )

            total_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        print("\033[32m train state: " + metric.value())

        # 验证阶段
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion)

        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    # 测试阶段
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion)

    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.4f} | best_test_{m}: {metric_value[m]:.4f}")

    return metric_value


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion):
    model.eval()
    metric = Metric(metrics)

    for samples, targets, _ in tqdm(data_loader, desc="Evaluating"):
        samples, targets = samples.to(device), targets.to(device)

        # 【核心对齐】：处理标签维度和类型
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        elif targets.dim() > 1:
            targets = targets.view(-1)
        targets = targets.long()

        out = model(samples, alpha=0.0) # 评估时 alpha 设为 0

        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        loss = criterion(logits, targets)
        _, pred = torch.max(logits, dim=1)
        metric.update(pred, targets, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values