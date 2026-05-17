from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.Models import Model
from models.PCL_TDGCN import Discriminator
from Trainer.PCL_TDGCNTraining import train, train_no_pcl
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import result_log, setup_seed, state_log


def _load_model_params():
    params = {
        "layers": 2,
        "hidden_1": 256,
        "hidden_2": 64,
        "weight_decay": 0.001,
    }
    param_path = Path(__file__).resolve().parent / "config" / "model_param" / "PCL_TDGCN.yaml"
    if param_path.exists():
        with open(param_path, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        params.update(loaded.get("params", {}))
    print(f"\nUsing setting from {param_path}\n")
    print("PCL_TDGCN Model, Parameters:")
    for key, value in params.items():
        print("{:30}{}".format(key + ":", value))
    return params


def _make_indexed_dataset(data, label):
    data = np.asarray(data)
    label = np.asarray(label)
    return torch.utils.data.TensorDataset(
        torch.Tensor(data),
        torch.arange(len(data)).long(),
        torch.Tensor(label),
    )


def _make_dataset(data, label):
    return torch.utils.data.TensorDataset(torch.Tensor(np.asarray(data)), torch.Tensor(np.asarray(label)))


def _test_subject_labels(test_indexes, test_data_by_subject):
    labels = []
    for subject_index, subject_data in zip(test_indexes, test_data_by_subject):
        labels.extend([subject_index + 1] * len(subject_data))
    return np.asarray(labels)


def _validate_supported_setting(args, setting):
    supported_models = {"PCL_TDGCN", "PCL_TDGCN_MLP", "PCL_TDGCN_NO_PCL"}
    supported_datasets = {
        "seed_de_lds": "seed_sub_independent_train_val_test_setting",
        "seediv_de_lds": "seediv_sub_independent_train_val_test_setting",
        "deap": "deap_sub_independent_train_val_test_setting",
    }
    if args.model not in supported_models:
        raise ValueError(f"PCL_TDGCN_train.py supports only -model {sorted(supported_models)}")
    if args.dataset not in supported_datasets:
        raise ValueError(
            f"PCL_TDGCN LibEER benchmark supports only -dataset {sorted(supported_datasets)}"
        )
    expected_setting = supported_datasets[args.dataset]
    if args.setting is not None and args.setting != expected_setting:
        raise ValueError(f"{args.dataset} requires -setting {expected_setting}")
    if setting.experiment_mode != "subject-independent" or setting.split_type != "train-val-test":
        raise ValueError(
            "PCL_TDGCN LibEER benchmark requires a subject-independent train-val-test setting"
        )
    if args.sample_length != 1:
        raise ValueError("PCL_TDGCN currently supports only -sample_length 1")
    if args.dataset == "deap":
        if args.label_used is None or len(args.label_used) != 1:
            raise ValueError("DEAP PCL_TDGCN smoke supports exactly one -label_used value")
        if args.bounds is None or len(args.bounds) != 2:
            raise ValueError("DEAP PCL_TDGCN requires -bounds LOW HIGH")


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    _validate_supported_setting(args, setting)
    setup_seed(args.seed)

    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    model_params = _load_model_params()
    best_metrics = []

    for data_i, label_i in zip(data, label):
        split_indexes = get_split_index(data_i, label_i, setting)
        for train_indexes, test_indexes, val_indexes in zip(
            split_indexes["train"], split_indexes["test"], split_indexes["val"]
        ):
            setup_seed(args.seed)
            print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")
            if val_indexes[0] == -1:
                raise ValueError("PCL_TDGCN requires a validation split for target-domain adaptation")

            _, _, _, _, test_data_by_subject, _ = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, keep_dim=True
            )
            test_sub_label = _test_subject_labels(test_indexes, test_data_by_subject)

            train_data, train_label, val_data, val_label, test_data, test_label = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, keep_dim=False
            )
            dataset_source = _make_indexed_dataset(train_data, train_label)
            dataset_target = _make_indexed_dataset(val_data, val_label)
            dataset_val = _make_dataset(val_data, val_label)
            dataset_test = _make_dataset(test_data, test_label)

            model = Model[args.model](
                in_planes=[feature_dim, channels],
                layers=model_params["layers"],
                hidden_1=model_params["hidden_1"],
                hidden_2=model_params["hidden_2"],
                num_of_class=num_classes,
                device=device,
                source_num=len(dataset_source),
                target_num=len(dataset_target),
            )
            domain_discriminator = Discriminator(model_params["hidden_2"])
            optimizer = optim.RMSprop(
                list(model.parameters()) + list(domain_discriminator.parameters()),
                lr=args.lr,
                weight_decay=model_params["weight_decay"],
            )

            output_dir = make_output_dir(args, args.model)
            train_fn = train_no_pcl if args.model == "PCL_TDGCN_NO_PCL" else train
            round_metric = train_fn(
                model=model,
                domain_discriminator=domain_discriminator,
                dataset_source=dataset_source,
                dataset_target=dataset_target,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                batch_size=args.batch_size,
                epochs=args.epochs,
                num_workers=args.num_workers,
                test_sub_label=test_sub_label,
                channels=channels,
                feature_dim=feature_dim,
            )
            best_metrics.append(round_metric)

    result_log(args, best_metrics)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    state_log(args)
    main(args)
