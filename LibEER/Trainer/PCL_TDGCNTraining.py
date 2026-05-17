import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from utils.metric import Metric, SubMetric
from utils.store import save_state


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.0005):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        targets = targets.long().view(-1)
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.epsilon / (inputs.size(-1) - 1.0)
        weight.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.epsilon)
        return (-weight * log_prob).sum(dim=-1).mean()


class WarmStartGradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, coeff=1.0):
        ctx.coeff = coeff
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.iter_num = 0

    def forward(self, inputs):
        coeff = 2.0 * (self.hi - self.lo) / (1.0 + math.exp(-self.alpha * self.iter_num / self.max_iters))
        coeff = coeff - (self.hi - self.lo) + self.lo
        if self.auto_step:
            self.iter_num += 1
        return WarmStartGradientReverseFunction.apply(inputs, coeff)


class DAANLoss(nn.Module):
    def __init__(self, domain_discriminator, reduction="mean", max_iter=1000):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, source_features, target_features, source_logits=None, target_logits=None):
        features = self.grl(torch.cat((source_features, target_features), dim=0))
        domain_pred = self.domain_discriminator(features)
        source_pred, target_pred = domain_pred.chunk(2, dim=0)
        source_label = torch.ones((source_features.size(0), 1), device=source_features.device)
        target_label = torch.zeros((target_features.size(0), 1), device=target_features.device)
        return 0.5 * (self.bce(source_pred, source_label) + self.bce(target_pred, target_label))


class StepwiseLR:
    def __init__(self, optimizer, init_lr=0.001, gamma=10.0, decay_rate=0.75, max_iter=1000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.iter_num = 0

    def step(self):
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** self.decay_rate
        for param_group in self.optimizer.param_groups:
            param_group.setdefault("lr_mult", 1.0)
            param_group["lr"] = lr * param_group["lr_mult"]
        self.iter_num += 1


def _labels_to_index(labels):
    if labels.dim() > 1:
        labels = torch.argmax(labels, dim=1)
    return labels.long().view(-1)


def _prepare_eeg_features(samples, channels, feature_dim):
    samples = samples.float()
    if samples.dim() == 2:
        return samples
    if samples.dim() != 3:
        raise ValueError(
            "PCL_TDGCN currently supports LibEER sample_length=1 feature batches "
            "with shape [batch, channels, feature_dim] or [batch, feature_dim, channels]."
        )
    if channels is None or feature_dim is None:
        raise ValueError("PCL_TDGCN trainer requires channels and feature_dim for 3D feature batches")
    if samples.size(1) == channels and samples.size(2) == feature_dim:
        samples = samples.permute(0, 2, 1).contiguous()
    elif samples.size(1) == feature_dim and samples.size(2) == channels:
        samples = samples.contiguous()
    else:
        raise ValueError(
            f"Unexpected PCL_TDGCN input shape: {tuple(samples.shape)}; expected "
            f"[batch, {channels}, {feature_dim}] or [batch, {feature_dim}, {channels}]"
        )
    return samples.reshape(samples.size(0), -1)


def _make_loader(dataset, batch_size, sampler, num_workers=4, drop_last=False):
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
    )


@torch.no_grad()
def _initialize_source_banks(model, source_dataset, device, batch_size, num_workers, channels, feature_dim):
    loader = _make_loader(source_dataset, batch_size, SequentialSampler(source_dataset), num_workers)
    model.eval()
    for samples, indexes, labels in loader:
        samples = _prepare_eeg_features(samples, channels, feature_dim).to(device)
        indexes = indexes.to(device)
        labels = _labels_to_index(labels).to(device)
        model.get_init_banks(samples, indexes, labels)


@torch.no_grad()
def _initialize_target_banks(model, target_dataset, device, batch_size, num_workers, channels, feature_dim):
    loader = _make_loader(target_dataset, batch_size, SequentialSampler(target_dataset), num_workers)
    model.eval()
    for samples, indexes, _ in loader:
        samples = _prepare_eeg_features(samples, channels, feature_dim).to(device)
        indexes = indexes.to(device)
        model.get_init_banks_tgt(samples, indexes)


def _predict_logits_and_features(model, samples):
    features, _ = model.encoder(samples)
    logits = model.cls_classifier(features)
    return logits, features


def _train_epoch(
    model, domain_loss, criterion, optimizer, source_loader, target_loader, device, epoch, epochs, channels, feature_dim
):
    model.train()
    domain_loss.train()

    target_iter = iter(target_loader)
    total_loss = 0.0
    batch_count = 0
    metric = Metric(["acc"])

    for source_samples, source_indexes, source_labels in tqdm(
        source_loader, total=len(source_loader), desc=f"Train Epoch {epoch + 1}/{epochs}", leave=False
    ):
        try:
            target_samples, target_indexes, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_samples, target_indexes, _ = next(target_iter)

        model.train()
        source_samples = _prepare_eeg_features(source_samples, channels, feature_dim).to(device)
        target_samples = _prepare_eeg_features(target_samples, channels, feature_dim).to(device)
        source_indexes = source_indexes.to(device)
        target_indexes = target_indexes.to(device)
        source_labels = _labels_to_index(source_labels).to(device)

        outputs = model(
            source_samples,
            target_samples,
            source_labels,
            source_indexes,
            target_indexes,
            epoch,
            epochs,
        )
        (
            source_logits,
            source_features,
            target_logits,
            target_features,
            _source_att,
            _target_att,
            _source_sim,
            target_sim,
            target_cluster_label,
            source_to_target_prob,
            target_to_source_prob,
            source_to_source_prob,
            target_to_target_prob,
        ) = outputs

        cls_loss = criterion(source_logits, source_labels)
        source_prob = F.softmax(source_logits, dim=1)
        max_prob, _ = source_prob.max(dim=1)
        source_mask = max_prob > 0.7
        if source_mask.any():
            source_loss = criterion(source_logits[source_mask], source_labels[source_mask])
        else:
            source_loss = torch.tensor(0.0, device=device)

        target_loss = criterion(target_sim, target_cluster_label.long())
        transfer_loss = domain_loss(
            source_features + 0.005 * torch.randn_like(source_features),
            target_features + 0.005 * torch.randn_like(target_features),
            source_prob,
            F.softmax(target_logits, dim=1),
        )

        boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-epoch / 1000.0)) - 1.0)
        cross_domain_loss = _entropy(source_to_target_prob) + _entropy(target_to_source_prob)
        in_domain_loss = _entropy(source_to_source_prob) + _entropy(target_to_target_prob)
        loss = cls_loss + transfer_loss + source_loss + boost_factor * target_loss
        loss = loss + 0.2 * (cross_domain_loss + in_domain_loss)

        if torch.isnan(loss).any():
            print(f"Warning: NaN loss detected at epoch {epoch}, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.update(torch.argmax(source_logits, dim=1), source_labels, loss.item())
        total_loss += loss.item()
        batch_count += 1

    print("\033[32m train state: " + metric.value())
    return total_loss / max(batch_count, 1)


def _train_epoch_no_pcl(
    model, domain_loss, criterion, optimizer, source_loader, target_loader, device, epoch, epochs, channels, feature_dim
):
    model.train()
    domain_loss.train()

    target_iter = iter(target_loader)
    total_loss = 0.0
    batch_count = 0
    metric = Metric(["acc"])

    for source_samples, _source_indexes, source_labels in tqdm(
        source_loader, total=len(source_loader), desc=f"Train Epoch {epoch + 1}/{epochs}", leave=False
    ):
        try:
            target_samples, _target_indexes, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_samples, _target_indexes, _ = next(target_iter)

        model.train()
        source_samples = _prepare_eeg_features(source_samples, channels, feature_dim).to(device)
        target_samples = _prepare_eeg_features(target_samples, channels, feature_dim).to(device)
        source_labels = _labels_to_index(source_labels).to(device)

        source_logits, source_features = _predict_logits_and_features(model, source_samples)
        target_logits, target_features = _predict_logits_and_features(model, target_samples)

        cls_loss = criterion(source_logits, source_labels)
        source_prob = F.softmax(source_logits, dim=1)
        max_prob, _ = source_prob.max(dim=1)
        source_mask = max_prob > 0.7
        if source_mask.any():
            source_loss = criterion(source_logits[source_mask], source_labels[source_mask])
        else:
            source_loss = torch.tensor(0.0, device=device)

        transfer_loss = domain_loss(
            source_features + 0.005 * torch.randn_like(source_features),
            target_features + 0.005 * torch.randn_like(target_features),
            source_prob,
            F.softmax(target_logits, dim=1),
        )
        loss = cls_loss + transfer_loss + source_loss

        if torch.isnan(loss).any():
            print(f"Warning: NaN loss detected at epoch {epoch}, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.update(torch.argmax(source_logits, dim=1), source_labels, loss.item())
        total_loss += loss.item()
        batch_count += 1

    print("\033[32m train state: " + metric.value())
    return total_loss / max(batch_count, 1)


def _entropy(prob):
    return -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean()


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, channels, feature_dim, sub_label_loader=None):
    model.eval()
    metric = SubMetric(metrics) if sub_label_loader is not None else Metric(metrics)
    if sub_label_loader is None:
        iterator = ((batch, None) for batch in data_loader)
        total = len(data_loader)
    else:
        iterator = zip(data_loader, sub_label_loader)
        total = len(data_loader)

    for (samples, labels), sub_labels in tqdm(iterator, total=total, desc="Evaluating : ", leave=False):
        samples = _prepare_eeg_features(samples, channels, feature_dim).to(device)
        labels = _labels_to_index(labels).to(device)
        logits = model.target_predict(samples)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        if sub_label_loader is None:
            metric.update(preds, labels, loss.item())
        else:
            metric.update(preds, labels, sub_labels.to(device), loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values


def train(
    model,
    domain_discriminator,
    dataset_source,
    dataset_target,
    dataset_val,
    dataset_test,
    device,
    output_dir,
    metrics=None,
    metric_choose=None,
    optimizer=None,
    batch_size=48,
    epochs=1000,
    num_workers=4,
    test_sub_label=None,
    channels=None,
    feature_dim=None,
):
    if metrics is None:
        metrics = ["acc"]
    if metric_choose is None:
        metric_choose = metrics[0]

    source_loader = _make_loader(
        dataset_source, batch_size, RandomSampler(dataset_source), num_workers, drop_last=True
    )
    target_loader = _make_loader(
        dataset_target, batch_size, RandomSampler(dataset_target), num_workers, drop_last=True
    )
    val_loader = _make_loader(dataset_val, batch_size, SequentialSampler(dataset_val), num_workers)
    test_loader = _make_loader(dataset_test, batch_size, SequentialSampler(dataset_test), num_workers)
    sub_label_loader = None
    if test_sub_label is not None:
        sub_label_tensor = torch.as_tensor(test_sub_label).long()
        sub_label_loader = DataLoader(
            sub_label_tensor,
            batch_size=batch_size,
            sampler=SequentialSampler(sub_label_tensor),
            num_workers=num_workers,
        )

    model = model.to(device)
    domain_discriminator = domain_discriminator.to(device)
    criterion = LabelSmoothingCrossEntropy().to(device)
    domain_loss = DAANLoss(domain_discriminator, max_iter=epochs).to(device)
    scheduler = StepwiseLR(optimizer, init_lr=optimizer.param_groups[0]["lr"], max_iter=epochs)

    _initialize_source_banks(model, dataset_source, device, batch_size, num_workers, channels, feature_dim)
    _initialize_target_banks(model, dataset_target, device, batch_size, num_workers, channels, feature_dim)

    best_metric = {name: float("-inf") for name in metrics}
    for epoch in range(epochs):
        _train_epoch(
            model, domain_loss, criterion, optimizer, source_loader, target_loader, device, epoch, epochs,
            channels, feature_dim
        )
        scheduler.step()
        metric_value = evaluate(model, val_loader, device, metrics, criterion, channels, feature_dim)
        for metric_name in metrics:
            if metric_value[metric_name] > best_metric[metric_name]:
                best_metric[metric_name] = metric_value[metric_name]
                save_state(output_dir, model, optimizer, epoch + 1, metric=metric_name)

    checkpoint = torch.load(f"{output_dir}/checkpoint-best{metric_choose}", map_location=device)
    model.load_state_dict(checkpoint["model"])
    metric_value = evaluate(model, test_loader, device, metrics, criterion, channels, feature_dim, sub_label_loader)
    for metric_name in metrics:
        print(f"best_val_{metric_name}: {best_metric[metric_name]:.2f}")
        print(f"best_test_{metric_name}: {metric_value[metric_name]:.2f}")
    return metric_value


def train_no_pcl(
    model,
    domain_discriminator,
    dataset_source,
    dataset_target,
    dataset_val,
    dataset_test,
    device,
    output_dir,
    metrics=None,
    metric_choose=None,
    optimizer=None,
    batch_size=48,
    epochs=1000,
    num_workers=4,
    test_sub_label=None,
    channels=None,
    feature_dim=None,
):
    if metrics is None:
        metrics = ["acc"]
    if metric_choose is None:
        metric_choose = metrics[0]

    source_loader = _make_loader(
        dataset_source, batch_size, RandomSampler(dataset_source), num_workers, drop_last=True
    )
    target_loader = _make_loader(
        dataset_target, batch_size, RandomSampler(dataset_target), num_workers, drop_last=True
    )
    val_loader = _make_loader(dataset_val, batch_size, SequentialSampler(dataset_val), num_workers)
    test_loader = _make_loader(dataset_test, batch_size, SequentialSampler(dataset_test), num_workers)
    sub_label_loader = None
    if test_sub_label is not None:
        sub_label_tensor = torch.as_tensor(test_sub_label).long()
        sub_label_loader = DataLoader(
            sub_label_tensor,
            batch_size=batch_size,
            sampler=SequentialSampler(sub_label_tensor),
            num_workers=num_workers,
        )

    model = model.to(device)
    domain_discriminator = domain_discriminator.to(device)
    criterion = LabelSmoothingCrossEntropy().to(device)
    domain_loss = DAANLoss(domain_discriminator, max_iter=epochs).to(device)
    scheduler = StepwiseLR(optimizer, init_lr=optimizer.param_groups[0]["lr"], max_iter=epochs)

    best_metric = {name: float("-inf") for name in metrics}
    for epoch in range(epochs):
        _train_epoch_no_pcl(
            model, domain_loss, criterion, optimizer, source_loader, target_loader, device, epoch, epochs,
            channels, feature_dim
        )
        scheduler.step()
        metric_value = evaluate(model, val_loader, device, metrics, criterion, channels, feature_dim)
        for metric_name in metrics:
            if metric_value[metric_name] > best_metric[metric_name]:
                best_metric[metric_name] = metric_value[metric_name]
                save_state(output_dir, model, optimizer, epoch + 1, metric=metric_name)

    checkpoint = torch.load(f"{output_dir}/checkpoint-best{metric_choose}", map_location=device)
    model.load_state_dict(checkpoint["model"])
    metric_value = evaluate(model, test_loader, device, metrics, criterion, channels, feature_dim, sub_label_loader)
    for metric_name in metrics:
        print(f"best_val_{metric_name}: {best_metric[metric_name]:.2f}")
        print(f"best_test_{metric_name}: {metric_value[metric_name]:.2f}")
    return metric_value
