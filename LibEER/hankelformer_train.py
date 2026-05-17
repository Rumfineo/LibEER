from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, get_split_index
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import result_log, setup_seed, sub_result_log
from Trainer.hankelformerTraining import train, InfoNCELoss # 确保从你的新训练脚本导入
from utils.store import make_output_dir

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np

from models.hankelformer import HankelFormer


class Configs:
    def __init__(self, seq_len, enc_in, num_classes, num_domains, window_size): # 增加 num_domains
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_classes = num_classes
        self.num_domains = num_domains # 新增：用于 DANN 判别器

        # hankel
        self.window_size = window_size
        # transformer
        self.d_model = 256
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 1024
        self.dropout = 0.1
        self.activation = "gelu"
        self.factor = 5
        self.output_attention = False
        self.embed = "timeF"
        self.freq = "s"
        self.use_norm = False


def make_x_mark(batch_size, seq_len, device, mark_dim=1):
    return torch.zeros(batch_size, seq_len, mark_dim, device=device)

class HankelWrapper(nn.Module):
    """
    修正：适配 DANN 的 alpha 和 domain_logits
    """
    def __init__(self, backbone: nn.Module, mark_dim=1):
        super().__init__()
        self.backbone = backbone
        self.mark_dim = mark_dim

    def forward(self, samples, alpha=1.0): # 增加 alpha 参数
        if samples.dim() == 4:
            samples = samples.squeeze(1)
        x_enc = samples.permute(0, 2, 1).contiguous()
        x_mark_enc = make_x_mark(x_enc.size(0), x_enc.size(1), x_enc.device, self.mark_dim)

        # 修正：接收 4 个返回值 (logits, domain_logits, repr1, repr2)
        return self.backbone(x_enc, x_mark_enc, alpha=alpha)


def index_to_data_with_subjects(data, label, subj_ids, train_indexes, test_indexes, val_indexes, keep_dim=False):
    train_original_ids = sorted({subj_ids[i] for i in train_indexes})
    id_map = {old_id: new_id for new_id, old_id in enumerate(train_original_ids)}

    train_data, train_label, train_subj = [], [], []
    val_data, val_label, val_subj = [], [], []
    test_data, test_label, test_subj = [], [], []

    def append_split(indexes, data_out, label_out, subj_out, is_train):
        if len(indexes) > 0 and indexes[0] == -1:
            return
        for index in indexes:
            samples = data[index]
            labels = label[index]
            domain = id_map[subj_ids[index]] if is_train else -1
            if keep_dim:
                data_out.append(samples)
                label_out.append(labels)
                subj_out.append(domain)
            else:
                data_out.extend(samples)
                label_out.extend(labels)
                subj_out.extend([domain] * len(samples))

    append_split(train_indexes, train_data, train_label, train_subj, True)
    append_split(val_indexes, val_data, val_label, val_subj, False)
    append_split(test_indexes, test_data, test_label, test_subj, False)

    if not keep_dim:
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        train_subj = np.array(train_subj)
        val_data = np.array(val_data)
        val_label = np.array(val_label)
        val_subj = np.array(val_subj)
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        test_subj = np.array(test_subj)

    return train_data, train_label, train_subj, val_data, val_label, val_subj, test_data, test_label, test_subj


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)

    data, label, channels, feature_dim, num_classes = get_data(setting)

    # 动态获取受试者总数
    # 如果 data 结构是 (session, subject, trial)，则：
    all_subject_ids = []
    # 遍历所有 session 收集所有受试者的原始 ID
    for s in range(len(data)):
        for sub_idx in range(len(data[s])):
            all_subject_ids.append(sub_idx)

    unique_subjs = np.unique(all_subject_ids)
    num_subjects = len(unique_subjs)
    print(f"Detected {num_subjects} subjects in total.")

    data, label = merge_to_part(data, label, setting)

    device = torch.device(args.device)
    best_metrics = []
    subjects_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        all_subj_ids = list(range(len(data_i)))

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
                zip(tts['train'], tts['test'], tts['val']), 1):

            setup_seed(args.seed)

            # 修正 1：index_to_data 返回 9 个值，解包受试者 ID
            train_data, train_label, train_subj, \
            val_data, val_label, val_subj, \
            test_data, test_label, test_subj = \
                index_to_data_with_subjects(data_i, label_i, subj_ids=all_subj_ids, train_indexes=train_indexes, test_indexes=test_indexes, val_indexes= val_indexes, keep_dim=args.keep_dim)

            if len(val_data) == 0:
                val_data, val_label, val_subj = test_data, test_label, test_subj

            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample')

            # 维度整理 [N, 1, C, L]
            train_data = np.transpose(train_data, (0, 2, 1))[:, np.newaxis, :, :]
            val_data   = np.transpose(val_data,   (0, 2, 1))[:, np.newaxis, :, :]
            test_data  = np.transpose(test_data,  (0, 2, 1))[:, np.newaxis, :, :]

            # 修正 2：计算训练集中受试者数量
            num_domains = len(set(train_subj))
            print("num_domains: ", num_domains)
            seq_len = train_data.shape[-1]
            enc_in = train_data.shape[-2]
            window_size = getattr(args, "window_size", 48)
            cfg = Configs(seq_len=seq_len, enc_in=enc_in, num_classes=num_classes, num_domains=num_domains, window_size=window_size)

            backbone = HankelFormer(cfg)
            model = HankelWrapper(backbone, mark_dim=1)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = StepLR(optimizer, gamma=0.3, step_size=100)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            contrastive_criterion = InfoNCELoss(temperature=getattr(args, "temp", 0.5))
            lambda_contrastive = getattr(args, "lambda_contrastive", 0.1)
            # 新增：领域对抗权重
            lambda_domain = getattr(args, "lambda_domain", 0.1)

            output_dir = make_output_dir(args, "HankelFormer")

            # 修正 3：按照新接口传入数据元组
            round_metric = train(
                model=model,
                train_tuple=(train_data, train_label, train_subj),
                val_tuple=(val_data, val_label, val_subj),
                test_tuple=(test_data, test_label, test_subj),
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=args.batch_size,
                epochs=args.epochs,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                lambda_contrastive=lambda_contrastive,
                lambda_domain=lambda_domain
            )

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
