import os
import sys
import argparse
import json
import re
import random
from typing import Dict, Tuple, OrderedDict, List, Mapping, Optional, Sequence

import torch
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from models.feature_vis import (
    append_gradient_flow_metrics,
    collect_gradient_flow_metrics,
    plot_gradient_flow_curves,
    retain_gradient_flow,
    save_gradient_flow_figure,
    visualize_sacon_step,
    log_sacon_gradcam,
)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AblationConfig, TrainConfig, get_ablation_configs
from models.sacon import SACON
from losses.losses import SACANLoss
from data.dataset import ONFHDataset
from utils.ssl_weights import load_ssl_encoder_weights, set_sacon_encoder_trainable
from utils.metrics import compute_binary_metrics, compute_stage_metrics, MetricTracker, plot_confusion_matrix, \
    plot_roc_curves, plot_precision_recall_curves, plot_binary_classification_metrics, plot_class_distribution, \
    ordinal_logits_to_class_probs, dca_curve, plot_dca, calibration_metrics, plot_calibration
from sklearn.metrics import roc_auc_score
import numpy as np


DEFAULT_DATA_ROOT = (
    r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH\uih_onfh_data_resampled.json"
)


def save_feature_clouds_on_demand(*args, **kwargs):
    from utils.dimensionality_reduction import save_feature_clouds

    return save_feature_clouds(*args, **kwargs)


def parse_real_test_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("real_test_ratio must be a number.") from exc
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError(
            f"real_test_ratio must satisfy 0 < ratio < 1, got {ratio}."
        )
    return ratio


def make_checkpoint_stem(task: str, ablation_name: str, run_tag: str = '') -> str:
    stem = f'{task}_{ablation_name}'
    if run_tag:
        stem = f'{stem}_{run_tag}'
    return stem


def get_base_patient_id(patient_id: str) -> str:
    return re.sub(r'_aug_\d+$', '', str(patient_id))


def gradient_flow_filename_metadata(
    batch: Mapping[str, object],
    task: str,
) -> Tuple[str, int, int]:
    patient_ids = batch.get("patient_id", "unknown")
    if isinstance(patient_ids, (list, tuple)):
        raw_patient_id = patient_ids[0] if patient_ids else "unknown"
    elif isinstance(patient_ids, torch.Tensor):
        raw_patient_id = patient_ids.flatten()[0].item() if patient_ids.numel() else "unknown"
    else:
        raw_patient_id = patient_ids

    file_id = re.sub(r'[<>:"/\\|?*\s]+', "_", str(raw_patient_id)).strip("._")
    file_id = (file_id or "unknown")[:80]

    if task == "bin":
        left_values = batch["left_binary"]
        right_values = batch["right_binary"]
    elif task == "stage":
        left_values = batch["label_left_ELS"]
        right_values = batch["label_right_ELS"]
    else:
        raise ValueError(f"未知任务：{task}")

    left_label = int(torch.as_tensor(left_values).flatten()[0].item())
    right_label = int(torch.as_tensor(right_values).flatten()[0].item())
    return file_id, left_label, right_label


def get_fold_stratify_labels(data: Dict[str, List], task: str) -> List[str]:
    if task == 'bin':
        return [
            f"{lb}_{rb}"
            for lb, rb in zip(data['label_left_bin'], data['label_right_bin'])
        ]
    return [
        f"{ls}_{rs}"
        for ls, rs in zip(data['label_left_ELS'], data['label_right_ELS'])
    ]


def is_synthetic_sample(patient_id: object) -> bool:
    return str(patient_id).startswith("gen_full_")


def get_stable_real_stratify_labels(
    data: Dict[str, List],
    task: str,
    real_indices: Sequence[int],
    min_class_count: int,
    max_class_count: Optional[int] = None,
) -> List[str]:
    combined = get_fold_stratify_labels(data, task)
    selected = [combined[index] for index in real_indices]
    counts = pd.Series(selected).value_counts().to_dict()
    combined_class_count_valid = (
        max_class_count is None or len(counts) <= max_class_count
    )
    if min(counts.values()) >= min_class_count and combined_class_count_valid:
        return selected

    if task == "bin":
        selected = [
            str(int(data["label_left_bin"][index]) + int(data["label_right_bin"][index]))
            for index in real_indices
        ]
        fallback_name = "positive-hip-count"
    else:
        selected = [
            str(max(int(data["label_left_ELS"][index]), int(data["label_right_ELS"][index])))
            for index in real_indices
        ]
        fallback_name = "maximum-bilateral-stage"
    fallback_counts = pd.Series(selected).value_counts().sort_index().to_dict()
    if min(fallback_counts.values()) < min_class_count:
        raise ValueError(
            f"降级分层后最小类别仍少于{min_class_count}例：{fallback_counts}"
        )
    if max_class_count is not None and len(fallback_counts) > max_class_count:
        raise ValueError(
            f"降级分层后类别数{len(fallback_counts)}仍超过划分容量"
            f"{max_class_count}：{fallback_counts}"
        )
    print(f"[CV] 双侧标签组合存在稀有类别，改用{fallback_name}分层：{fallback_counts}")
    return selected


def build_real_only_test_splits(
    data: Dict[str, List],
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
    real_test_ratio: float = 0.3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not 0.0 < real_test_ratio < 1.0:
        raise ValueError(
            f"real_test_ratio must satisfy 0 < ratio < 1, got {real_test_ratio}."
        )
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}.")

    patient_ids = list(data["patient_id"])
    all_labels = get_fold_stratify_labels(data, task)
    synthetic_indices = np.asarray(
        [index for index, patient_id in enumerate(patient_ids) if is_synthetic_sample(patient_id)],
        dtype=np.int64,
    )
    real_indices = np.asarray(
        [index for index, patient_id in enumerate(patient_ids) if not is_synthetic_sample(patient_id)],
        dtype=np.int64,
    )
    if len(real_indices) < 2:
        raise ValueError(
            f"真实样本数{len(real_indices)}少于2，无法划分训练集和测试集。"
        )

    splitter = ShuffleSplit(
        n_splits=n_splits,
        test_size=real_test_ratio,
        random_state=random_state,
    )
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id, (real_train_position, real_test_position) in enumerate(
        splitter.split(real_indices)
    ):
        train_index = np.concatenate(
            [real_indices[real_train_position], synthetic_indices]
        ).astype(np.int64)
        test_index = real_indices[real_test_position].astype(np.int64)
        if any(is_synthetic_sample(patient_ids[index]) for index in test_index):
            raise RuntimeError(f"fold {fold_id}测试集中检测到合成样本。")
        if set(synthetic_indices.tolist()) - set(train_index.tolist()):
            raise RuntimeError(f"fold {fold_id}训练集未包含全部合成样本。")
        if set(train_index.tolist()) & set(test_index.tolist()):
            raise RuntimeError(f"fold {fold_id}训练集与测试集存在样本重叠。")

        train_real_count = int(sum(not is_synthetic_sample(patient_ids[index]) for index in train_index))
        train_synthetic_count = int(len(train_index) - train_real_count)
        actual_real_test_ratio = len(test_index) / len(real_indices)
        print(
            f"[Fold {fold_id}] split_mode=RepeatedRealHoldout(data-level) | "
            f"target_real_test_ratio={real_test_ratio:.4f} "
            f"actual_real_test_ratio={actual_real_test_ratio:.4f} | "
            f"train_real={train_real_count} train_synthetic={train_synthetic_count} "
            f"test_real={len(test_index)} test_synthetic=0"
        )
        print(
            f"[Fold {fold_id}] test_label_counts="
            f"{pd.Series([all_labels[index] for index in test_index]).value_counts().sort_index().to_dict()}"
        )
        splits.append((train_index, test_index))
    return splits


def save_fold_split_csv(
    data: Dict[str, List],
    train_index: Sequence[int],
    test_index: Sequence[int],
    fold_id: int,
    fold_dir: str,
) -> str:
    sample_count = len(data["patient_id"])
    sample_columns = [
        key
        for key, values in data.items()
        if isinstance(values, (list, tuple)) and len(values) == sample_count
    ]
    rows: List[Dict[str, object]] = []
    for split_name, indices in (("train", train_index), ("test", test_index)):
        for index in indices:
            source_index = int(index)
            patient_id = data["patient_id"][source_index]
            row: Dict[str, object] = {
                "fold": int(fold_id),
                "split": split_name,
                "source_index": source_index,
                "sample_type": "synthetic" if is_synthetic_sample(patient_id) else "real",
            }
            row.update({key: data[key][source_index] for key in sample_columns})
            rows.append(row)

    split_table = pd.DataFrame(rows)
    output_path = os.path.join(fold_dir, "fold_data_split.csv")
    split_table.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"[Fold {fold_id}] 划分信息已保存：{output_path} | "
        f"train={len(train_index)} test={len(test_index)}"
    )
    return output_path


def validate_base_id_labels(data: Dict[str, List], base_ids: List[str]) -> None:
    labels_by_base_id = {}
    for idx, base_id in enumerate(base_ids):
        labels = (
            data['label_left_bin'][idx],
            data['label_right_bin'][idx],
            data['label_left_ELS'][idx],
            data['label_right_ELS'][idx],
        )
        if base_id in labels_by_base_id and labels_by_base_id[base_id] != labels:
            raise ValueError(
                f"Inconsistent labels for base patient ID {base_id}: "
                f"{labels_by_base_id[base_id]} vs {labels}"
            )
        labels_by_base_id[base_id] = labels


def print_fold_summary(
    fold_id: int,
    train_index,
    test_index,
    patient_ids: List[str],
    base_ids: List[str],
    stratify_labels: List[str],
    group_by_base_id: bool,
) -> None:
    train_base_ids = {base_ids[i] for i in train_index}
    test_base_ids = {base_ids[i] for i in test_index}
    overlap = train_base_ids & test_base_ids
    if group_by_base_id and overlap:
        raise ValueError(f"Fold {fold_id} has leaked base patient IDs: {sorted(overlap)[:10]}")

    train_aug = sum(1 for i in train_index if re.search(r'_aug_\d+$', str(patient_ids[i])))
    test_aug = sum(1 for i in test_index if re.search(r'_aug_\d+$', str(patient_ids[i])))
    train_label_counts = pd.Series([stratify_labels[i] for i in train_index]).value_counts().sort_index().to_dict()
    test_label_counts = pd.Series([stratify_labels[i] for i in test_index]).value_counts().sort_index().to_dict()
    labels_by_base_id = {}
    for idx, base_id in enumerate(base_ids):
        labels_by_base_id.setdefault(base_id, stratify_labels[idx])
    train_base_label_counts = pd.Series(
        [labels_by_base_id[base_id] for base_id in sorted(train_base_ids)]
    ).value_counts().sort_index().to_dict()
    test_base_label_counts = pd.Series(
        [labels_by_base_id[base_id] for base_id in sorted(test_base_ids)]
    ).value_counts().sort_index().to_dict()

    print(
        f"[Fold {fold_id}] train_samples={len(train_index)} val_samples={len(test_index)} | "
        f"train_base_ids={len(train_base_ids)} val_base_ids={len(test_base_ids)} | "
        f"train_aug={train_aug} val_aug={test_aug}"
    )
    print(f"[Fold {fold_id}] train_label_counts={train_label_counts}")
    print(f"[Fold {fold_id}] val_label_counts={test_label_counts}")
    print(f"[Fold {fold_id}] train_base_label_counts={train_base_label_counts}")
    print(f"[Fold {fold_id}] val_base_label_counts={test_base_label_counts}")


def build_cv_splits(
    data: Dict[str, List],
    task: str,
    group_by_base_id: bool,
    repeated_holdout_7_3: bool = False,
    real_only_test: bool = False,
    real_test_ratio: float = 0.3,
):
    patient_ids = list(data['patient_id'])
    if real_only_test:
        if group_by_base_id or repeated_holdout_7_3:
            raise ValueError(
                "real_only_test是数据层面重复留出，不能与group_by_base_id或"
                "repeated_holdout_7_3同时启用。"
            )
        return build_real_only_test_splits(
            data,
            task,
            real_test_ratio=real_test_ratio,
        )
    if repeated_holdout_7_3:
        base_ids = [get_base_patient_id(pid) for pid in patient_ids]
        stratify_labels = get_fold_stratify_labels(data, task)
        validate_base_id_labels(data, base_ids)

        group_indices = {}
        group_labels = {}
        for sample_idx, (base_id, label) in enumerate(zip(base_ids, stratify_labels)):
            group_indices.setdefault(base_id, []).append(sample_idx)
            group_labels.setdefault(base_id, label)

        unique_base_ids = list(group_indices)
        unique_labels = [group_labels[base_id] for base_id in unique_base_ids]
        label_counts = pd.Series(unique_labels).value_counts().sort_index().to_dict()
        print(
            "[CV] split_mode=RepeatedStratifiedGroupHoldout(base-patient-id) | "
            "train_ratio=0.70 val_ratio=0.30 repeats=5"
        )
        print(f"[CV] base_label_counts={label_counts}")

        splitter = StratifiedShuffleSplit(
            n_splits=5,
            test_size=0.3,
            random_state=42,
        )
        try:
            group_splits = list(splitter.split(unique_base_ids, unique_labels))
        except ValueError as exc:
            raise ValueError(
                "Cannot build 5 repeated stratified 70/30 holdout splits at the base-patient-ID level. "
                f"base_label_counts={label_counts}. Original error: {exc}"
            ) from exc

        splits = []
        for train_group_index, test_group_index in group_splits:
            train_base_ids = {unique_base_ids[i] for i in train_group_index}
            test_base_ids = {unique_base_ids[i] for i in test_group_index}
            train_index = np.asarray(
                [
                    idx
                    for base_id in unique_base_ids
                    if base_id in train_base_ids
                    for idx in group_indices[base_id]
                ],
                dtype=np.int64,
            )
            test_index = np.asarray(
                [
                    idx
                    for base_id in unique_base_ids
                    if base_id in test_base_ids
                    for idx in group_indices[base_id]
                ],
                dtype=np.int64,
            )
            splits.append((train_index, test_index))

        for fold_id, (train_index, test_index) in enumerate(splits):
            print_fold_summary(
                fold_id,
                train_index,
                test_index,
                patient_ids,
                base_ids,
                stratify_labels,
                group_by_base_id=True,
            )
        return splits

    if not group_by_base_id:
        print("[CV] split_mode=KFold(sample-index)")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return list(kf.split(patient_ids))

    base_ids = [get_base_patient_id(pid) for pid in patient_ids]
    stratify_labels = get_fold_stratify_labels(data, task)
    validate_base_id_labels(data, base_ids)

    print("[CV] split_mode=StratifiedGroupKFold(base-patient-id)")
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(patient_ids, stratify_labels, groups=base_ids))
    except ValueError as exc:
        print(f"[CV] StratifiedGroupKFold failed: {exc}")
        print("[CV] Falling back to GroupKFold(base-patient-id).")
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(patient_ids, stratify_labels, groups=base_ids))

    for fold_id, (train_index, test_index) in enumerate(splits):
        print_fold_summary(
            fold_id,
            train_index,
            test_index,
            patient_ids,
            base_ids,
            stratify_labels,
            group_by_base_id=True,
        )
    return splits


def move_training_batch_to_device(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    images = (
        batch['left_t1'].to(device, non_blocking=non_blocking),
        batch['left_t2'].to(device, non_blocking=non_blocking),
        batch['right_t1'].to(device, non_blocking=non_blocking),
        batch['right_t2'].to(device, non_blocking=non_blocking),
    )
    labels = {
        'left_binary': batch['left_binary'].to(device, non_blocking=non_blocking),
        'left_stage': batch['label_left_ELS'].to(device, non_blocking=non_blocking),
        'right_binary': batch['right_binary'].to(device, non_blocking=non_blocking),
        'right_stage': batch['label_right_ELS'].to(device, non_blocking=non_blocking),
    }
    return images, labels


def train_one_epoch(
    model: SACON,
    loader: data_utils.DataLoader,
    criterion: SACANLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    total_steps: int,
    writer: SummaryWriter,
    epoch: int,
    scaler: GradScaler,
    vis_interval=0,
    progress_update_interval=20,
    ordinal=False,
    task = None,
    fold_save=None,
    grad_flow_interval=0,
    grad_flow_samples=1,
) -> Dict[str, float]:
    model.train()
    total_loss = torch.zeros((), device=device)
    loss_sums = {
        key: torch.zeros((), device=device)
        for key in ('binary', 'stage', 'registration', 'decouple', 'concentration')
    }

    loop = tqdm(loader,total=len(loader))
    grad_flow_logged = 0
    enable_grad_flow = (
        fold_save is not None
        and grad_flow_interval is not None
        and grad_flow_interval > 0
        and epoch % grad_flow_interval == 0
    )

    for batch_idx, batch in enumerate(loop, start=1):
        total_steps += 1
        (xl_t1, xl_t2, xr_t1, xr_t2), labels = move_training_batch_to_device(
            batch, device
        )


        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)
            log_grad_flow = enable_grad_flow and grad_flow_logged < max(1, grad_flow_samples)
            if log_grad_flow:
                retain_gradient_flow(outputs)
            loss, loss_dict = criterion(outputs, labels)


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if log_grad_flow:
            metrics_path = os.path.join(fold_save, "gradient_interaction_metrics.csv")
            row = collect_gradient_flow_metrics(outputs, epoch, total_steps)
            append_gradient_flow_metrics(row, metrics_path)
            suffix = f"epoch_{epoch:03d}" if grad_flow_logged == 0 else f"epoch_{epoch:03d}_step_{total_steps}"
            file_id, left_label, right_label = gradient_flow_filename_metadata(batch, task)
            suffix = f"{suffix}_id_{file_id}_L{left_label}_R{right_label}"
            save_gradient_flow_figure(
                outputs,
                os.path.join(fold_save, f"gradient_flow_{suffix}.png"),
                max_samples=1,
                images={
                    "left_t1": xl_t1,
                    "left_t2": xl_t2,
                    "right_t1": xr_t1,
                    "right_t2": xr_t2,
                    "left_fusion": outputs.get("f_left"),
                    "right_fusion": outputs.get("f_right"),
                },
            )
            plot_gradient_flow_curves(
                metrics_path,
                os.path.join(fold_save, "gradient_interaction_curves.png"),
            )
            grad_flow_logged += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()
        for k in loss_sums:
            component = loss_dict.get(k)
            if component is not None:
                loss_sums[k] += component.detach() if isinstance(component, torch.Tensor) else component

        if progress_update_interval and batch_idx % progress_update_interval == 0:
            task_key = 'binary' if task == 'bin' else 'stage'
            display_keys = ('total_loss', task_key, 'registration', 'decouple', 'concentration')
            display_tensors = [loss.detach()] + [loss_dict[key].detach() for key in display_keys[1:]]
            display_values = torch.stack(display_tensors).float().cpu().tolist()
            loop.set_postfix(dict(zip(display_keys, display_values)))

        if vis_interval and vis_interval > 0 and total_steps % vis_interval == 0:
            visualize_sacon_step(
                model=model,
                outputs=outputs,
                xl_t1=xl_t1,
                xl_t2=xl_t2,
                xr_t1=xr_t1,
                xr_t2=xr_t2,
                writer=writer,
                global_step=total_steps,
            )

    n = len(loader)
    averages = {
        'total': float((total_loss / n).item()),
        **{k: float((v / n).item()) for k, v in loss_sums.items()},
    }
    writer.add_scalar('train/total_loss', averages['total'], epoch)
    writer.add_scalar('train/decouple_loss', averages["decouple"], epoch)
    writer.add_scalar('train/reg_loss', averages["registration"], epoch)

    if task == 'bin':
        writer.add_scalar('train/bin_loss', averages["binary"], epoch)
    if task == 'stage':
        writer.add_scalar('train/stage_loss', averages["stage"], epoch)


    return averages


def save_binary_validation_artifacts(y_true, y_pred, y_prob, fold_save, epoch):
    plot_binary_classification_metrics(y_true, y_pred, y_prob, fold_save, epoch)

    thresholds, nb_model, nb_all, nb_none = dca_curve(y_true, y_prob)
    fig_dca = plot_dca(
        thresholds,
        nb_model,
        nb_all,
        nb_none,
        title="Decision Curve",
        path=os.path.join(fold_save, 'DCA_binary_classification.png'),
    )
    pd.DataFrame({
        'thresholds': thresholds,
        'nb_model': nb_model,
        'nb_all': nb_all,
        'nb_none': nb_none,
    }).to_csv(os.path.join(fold_save, 'DCA_binary.csv'), index=False, mode='a', header=True)
    plt.close(fig_dca)

    cal = calibration_metrics(y_true, y_prob, n_bins=10, strategy="uniform")
    fig_cal = plot_calibration(
        cal["prob_true"],
        cal["prob_pred"],
        title="Calibration",
        path=os.path.join(fold_save, 'calibration_binary_classification.png'),
    )
    pd.DataFrame({
        'prob_true': cal["prob_true"],
        'prob_pred': cal["prob_pred"],
    }).to_csv(
        os.path.join(fold_save, 'calibration_binary.csv'),
        mode='a',
        index=False,
        header=True,
    )
    plt.close(fig_cal)


@torch.no_grad()
def validate(
    model: SACON,
    loader: data_utils.DataLoader,
    criterion: SACANLoss,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_stages: int,
    class_names: List[str],
    ordinal = False,
    fold_save = None,
    every_save_time=None,
    task=None,
    threshold=0.5,
    fold_id=1,
    run_tag: str = '',
    model_variant: str = '',
):
    model.eval()
    total_loss = torch.zeros((), device=device)
    bin_loss = torch.zeros((), device=device)
    stage_loss = torch.zeros((), device=device)
    decouple_loss = torch.zeros((), device=device)
    reg_loss = torch.zeros((), device=device)

    if task == 'stage':
        tracker = MetricTracker(num_stages, class_names,ordinal=ordinal)
    if task == 'bin':
        all_lb_prob, all_lb_true = [], []
        all_rb_prob, all_rb_true = [], []

        left_features = []
        right_features = []
        left_labels = []
        right_labels = []

    loop = tqdm(loader,total=len(loader))
    for batch in loop:
        (xl_t1, xl_t2, xr_t1, xr_t2), labels = move_training_batch_to_device(
            batch, device
        )
        with autocast(enabled=True):
            outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)
            loss, other_loss = criterion(outputs, labels)

            total_loss += loss.detach()
            decouple_loss += other_loss["decouple"]
            reg_loss += other_loss["registration"]

            if task == 'bin':
                bin_loss += other_loss["binary"]
            if task == 'stage':
                stage_loss += other_loss["stage"]


        if task=='stage':
            batch_loss = float(loss.detach().item())
            tracker.update(batch_loss, outputs["left_stage"], labels["left_stage"], xl_t1.size(0), sample_ids=batch.get('patient_id'))
            tracker.update(batch_loss,outputs["right_stage"],labels["right_stage"],xr_t1.size(0), sample_ids=batch.get('patient_id'))

        if task == 'bin':
            all_lb_prob.extend(torch.sigmoid(outputs['left_binary']).squeeze(1).cpu().tolist())
            all_lb_true.extend(batch['left_binary'].tolist())
            all_rb_prob.extend(torch.sigmoid(outputs['right_binary']).squeeze(1).cpu().tolist())
            all_rb_true.extend(batch['right_binary'].tolist())


    total_loss = float(total_loss.item())
    decouple_loss = float(decouple_loss.item())
    reg_loss = float(reg_loss.item())
    bin_loss = float(bin_loss.item())
    stage_loss = float(stage_loss.item())
    writer.add_scalar('val/total_loss', total_loss / len(loader), epoch)
    writer.add_scalar('val/decouple_loss', decouple_loss / len(loader), epoch)
    writer.add_scalar('val/reg_loss', reg_loss / len(loader), epoch)

    if task == 'bin':
        writer.add_scalar('val/bin_loss', bin_loss / len(loader), epoch)
        try:
            left_auc  = roc_auc_score(all_lb_true, all_lb_prob) if len(set(all_lb_true)) > 1 else 0.5
            right_auc = roc_auc_score(all_rb_true, all_rb_prob) if len(set(all_rb_true)) > 1 else 0.5
        except Exception:
            left_auc = right_auc = 0.5
        avg_auc = (left_auc + right_auc) / 2.0
        y_true = np.concatenate((np.array(all_lb_true),np.array(all_rb_true)))
        y_pred = np.concatenate(((np.array(all_lb_prob)>=0.5).astype(int),(np.array(all_rb_prob)>=0.5).astype(int)))
        y_prob = np.concatenate((np.array(all_lb_prob),np.array(all_rb_prob)))

        save_origin_data = {'fold_id':np.array([fold_id]*len(y_true)),'epoch':np.array([epoch]*len(y_true)),'labels':y_true,'preds':y_pred,'probs':y_prob}
        pd.DataFrame(save_origin_data).to_csv('bin_origin_data.csv', mode='a',index=False)

        other_m = compute_binary_metrics(y_true, y_prob)
        save_metrics = {'epoch':epoch, 'run_tag': run_tag, 'model_variant': model_variant, **other_m}
        bin_metrics_path = os.path.join(fold_save, 'bin_metrics.csv')
        pd.DataFrame([save_metrics]).to_csv(
            bin_metrics_path,
            mode='a',
            index=False,
            header=not os.path.exists(bin_metrics_path),
        )

        writer.add_scalar('val/acc', other_m['accuracy'], epoch)

        return {
            'val_loss': total_loss / len(loader),
            'left_auc': left_auc,
            'right_auc': right_auc,
            'avg_auc': avg_auc,
            '_artifact_data': (y_true, y_pred, y_prob),
        }

    if task=='stage':
        writer.add_scalar('val/stage_loss', stage_loss / len(loader), epoch)


        return {'val_loss': total_loss / len(loader)}, tracker


def run_experiment(
    ablation_name: str,
    train_cfg: TrainConfig,
    siemens_finetune_ckpt: Optional[str] = None,
) -> Tuple[float, str]:
    configs = get_ablation_configs()
    if ablation_name not in configs:
        raise ValueError(f"未知消融配置: {ablation_name}，可选: {list(configs.keys())}")

    ablation_cfg = configs[ablation_name]
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
    random.seed(train_cfg.random_seed)
    np.random.seed(train_cfg.random_seed)
    torch.manual_seed(train_cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.random_seed)

    print(f"  消融实验: {ablation_name}")
    print(f"  设备: {device}  | Epochs: {train_cfg.num_epochs}  | BS: {train_cfg.batch_size}")
    print(f"  配置: {vars(ablation_cfg)}")

    print("data_root:",train_cfg.data_root)
    with open(train_cfg.data_root, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    if train_cfg.online_train_augment and any(re.search(r'_aug_\d+$', str(pid)) for pid in data['patient_id']):
        print(
            "[WARN] online_train_augment is enabled but data_root contains offline augmented IDs. "
            "Prefer an original-only manifest, e.g. generated with pre_processed_data.py --exclude_aug."
        )

    cv_splits = build_cv_splits(
        data,
        train_cfg.task,
        train_cfg.group_by_base_id,
        train_cfg.repeated_holdout_7_3,
        train_cfg.real_only_test,
        train_cfg.real_test_ratio,
    )

    completed = []
    for fold_id, (train_index, test_index) in enumerate(cv_splits):
        if train_cfg.selected_folds is not None and fold_id not in train_cfg.selected_folds:
            continue
        print(train_index, test_index)
        fold_root = train_cfg.experiment_root or "."
        fold_dir = os.path.join(fold_root, "checkpoints", ablation_name, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        save_fold_split_csv(
            data=data,
            train_index=train_index,
            test_index=test_index,
            fold_id=fold_id,
            fold_dir=fold_dir,
        )

        train_ds = ONFHDataset(
            data_root=train_cfg.data_root,
            ids = train_index,
            volume_size=train_cfg.volume_size,
            augment=train_cfg.augment,
            online_train_augment=train_cfg.online_train_augment,
            late_aug_copies=train_cfg.late_aug_copies,
            nonlate_aug_copies=train_cfg.nonlate_aug_copies,
            late_aug_max_retries=train_cfg.late_aug_max_retries,
        )
        class_names = list(train_ds.CLASS_MAP.keys())
        plot_class_distribution(
            train_ds.left_stage_class_counts,
            class_names,
            save_path=os.path.join(fold_dir, "left_stage_class_distribution_train.png"),
            title="Train Class Distribution for Left Hip",
        )
        plot_class_distribution(
            train_ds.right_stage_class_counts,
            class_names,
            save_path=os.path.join(fold_dir, "right_stage_class_distribution_train.png"),
            title="Train Class Distribution for Right Hip",
        )
        val_ds   = ONFHDataset(
            data_root=train_cfg.data_root,
            ids = test_index,
            volume_size=train_cfg.volume_size,
            augment=False,
            online_train_augment=False,
        )

        real_train_indices = [
            int(index)
            for index in train_index
            if not is_synthetic_sample(data["patient_id"][int(index)])
        ]
        left_stage_counts = pd.Series(
            [data["label_left_ELS"][index] for index in real_train_indices]
        ).value_counts().to_dict()
        right_stage_counts = pd.Series(
            [data["label_right_ELS"][index] for index in real_train_indices]
        ).value_counts().to_dict()
        left_bin_counts = pd.Series(
            [data["label_left_bin"][index] for index in real_train_indices]
        ).value_counts().to_dict()
        right_bin_counts = pd.Series(
            [data["label_right_bin"][index] for index in real_train_indices]
        ).value_counts().to_dict()

        def sqrt_inverse_weights(class_counts, num_classes):
            counts = np.asarray(
                [class_counts.get(class_index, 1) for class_index in range(num_classes)],
                dtype=np.float64,
            )
            weights = 1.0 / np.sqrt(counts)
            weights = weights / weights.sum() * num_classes
            return torch.tensor(weights, dtype=torch.float32)

        stage_class_weights = (
            sqrt_inverse_weights(left_stage_counts, train_cfg.num_stages),
            sqrt_inverse_weights(right_stage_counts, train_cfg.num_stages),
        )

        bilateral_negative_count = (
            left_bin_counts.get(0, 0) + right_bin_counts.get(0, 0)
        )
        bilateral_positive_count = (
            left_bin_counts.get(1, 0) + right_bin_counts.get(1, 0)
        )
        binary_pos_weight = max(bilateral_negative_count, 1) / max(
            bilateral_positive_count,
            1,
        )
        bin_class_weights = (
            torch.tensor([1.0, binary_pos_weight], dtype=torch.float32),
            torch.tensor([1.0, binary_pos_weight], dtype=torch.float32),
        )

        loader_kwargs = {
            'batch_size': train_cfg.batch_size,
            'num_workers': train_cfg.num_workers,
            'pin_memory': train_cfg.pin_memory and device.type == 'cuda',
        }
        if train_cfg.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=train_cfg.persistent_workers,
                prefetch_factor=train_cfg.prefetch_factor,
            )
        train_loader = data_utils.DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = data_utils.DataLoader(val_ds, shuffle=False, **loader_kwargs)

        model = SACON(ablation_cfg, base_channels=train_cfg.base_channels,
                      num_stages=train_cfg.num_stages,ordinal=train_cfg.ordinal,task=train_cfg.task).to(device)
        use_feature_hooks = (
            train_cfg.enable_cam_hooks
            or train_cfg.tensorboard_image_interval > 0
        )
        if use_feature_hooks:
            model.register_cam_hooks()
        if siemens_finetune_ckpt:
            checkpoint = torch.load(siemens_finetune_ckpt, map_location=device)
            source_train_cfg = checkpoint.get("train_cfg", {})
            source_task = source_train_cfg.get("task")
            if source_task and source_task != train_cfg.task:
                raise ValueError(
                    f"SIEMENS checkpoint task is {source_task}, but current task is {train_cfg.task}."
                )
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            ssl_load_info = {
                "siemens_finetune": {
                    "path": siemens_finetune_ckpt,
                    "initialization": "full_supervised_model",
                }
            }
            print(f"[SIEMENS fine-tune] Loaded full supervised model from {siemens_finetune_ckpt}")
        else:
            ssl_load_info = load_ssl_encoder_weights(
                model,
                t1_ckpt=train_cfg.ssl_t1_ckpt,
                t2_ckpt=train_cfg.ssl_t2_ckpt,
                device=device,
                strict=False,
            )
        if train_cfg.freeze_encoder_epochs > 0:
            set_sacon_encoder_trainable(model, False)
            print(f"[SSL] Encoders frozen for first {train_cfg.freeze_encoder_epochs} epoch(s).")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        print("stage_class_weights:", stage_class_weights)
        print("bin_class_weights:", bin_class_weights)

        criterion = SACANLoss(
            ablation_cfg,
            alpha_bin=train_cfg.alpha_bin,
            alpha_stage=train_cfg.alpha_stage,
            alpha_reg=train_cfg.alpha_reg,
            alpha_conc=train_cfg.alpha_conc,
            alpha_local_conc=train_cfg.alpha_local_conc,
            alpha_decouple=train_cfg.alpha_decouple,
            num_stages=train_cfg.num_stages,
            ordinal=train_cfg.ordinal,
            stage_class_weights=stage_class_weights,
            bin_class_weights = bin_class_weights,
            task=train_cfg.task,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        scaler = GradScaler(enabled=device.type == 'cuda')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.num_epochs
        )

        os.makedirs(train_cfg.save_dir, exist_ok=True)
        ckpt_stem = make_checkpoint_stem(train_cfg.task, ablation_name, train_cfg.run_tag)
        ckpt_path = os.path.join(fold_dir, f'{ckpt_stem}_best.pth')

        history = []
        model_variant = f"{train_cfg.run_tag}_{ablation_name}" if train_cfg.run_tag else ablation_name
        feature_cloud_interval = (
            train_cfg.every_save_time
            if train_cfg.feature_cloud_interval is None
            else train_cfg.feature_cloud_interval
        )
        grad_flow_interval = (
            train_cfg.every_save_time
            if train_cfg.grad_flow_interval is None
            else train_cfg.grad_flow_interval
        )

        total_step = 0
        writer = SummaryWriter(log_dir=os.path.join(fold_dir, "tensorboard"))
        
        def return_stage_metrics(tracker,epoch):
            cm = tracker.get_confusion_matrix()
            lbs, prbs = tracker.get_probabilities()
            plot_confusion_matrix(
                cm, list(train_cfg.class_names),
                fold_dir + "/" +"val_confusion_matrix.png",
                title=f"Val Confusion Matrix (Epoch {epoch})",
                epoch=epoch,
                fold_dir=fold_dir,
            )
            plot_roc_curves(
                lbs, prbs, list(train_cfg.class_names),
                fold_dir + "/" + "val_roc_curves.png",
                epoch=epoch,
                fold_dir=fold_dir,
            )
            plot_precision_recall_curves(
                lbs, prbs, list(train_cfg.class_names),
                fold_dir + "/" + "val_pr_curves.png",
                epoch=epoch,
                fold_dir=fold_dir,
            )

        best_val_loss = float('inf')
        best_val_accuracy = float('-inf')
        counter = 0
        start_epoch = 1
        if getattr(train_cfg, "resume_from_checkpoint", False) and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_loss = float(checkpoint.get("val_metrics", {}).get("val_loss", best_val_loss))
            if train_cfg.task == 'stage':
                best_val_accuracy = float(
                    checkpoint.get("val_metrics", {}).get("accuracy", best_val_accuracy)
                )
            print(f"[resume] fold={fold_id}: loaded {ckpt_path}; continuing at epoch {start_epoch}")

        train_range = range(start_epoch, train_cfg.num_epochs + 1)
        train_loop = tqdm(train_range, total=len(train_range))

        for epoch in train_loop:
            if train_cfg.freeze_encoder_epochs > 0 and epoch == train_cfg.freeze_encoder_epochs + 1:
                set_sacon_encoder_trainable(model, True)
                print(f"[SSL] Encoders unfrozen at epoch {epoch}.")

            total_step += len(train_loader)
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, total_step, writer, epoch,
                scaler=scaler,
                vis_interval=train_cfg.tensorboard_image_interval,
                progress_update_interval=train_cfg.progress_update_interval,
                ordinal=train_cfg.ordinal, task=train_cfg.task, fold_save=fold_dir,
                grad_flow_interval=grad_flow_interval,
                grad_flow_samples=train_cfg.grad_flow_samples,
            )
            binary_artifact_data = None
            val_tracker = None
            if train_cfg.task == 'bin':
                val_metrics = validate(model, val_loader, criterion, device, writer, epoch,
                                                    train_cfg.num_stages, list(train_cfg.class_names),
                                                    ordinal=train_cfg.ordinal, fold_save=fold_dir,
                                                    every_save_time=train_cfg.every_save_time,
                                                    task=train_cfg.task,fold_id=fold_id,
                                                    run_tag=train_cfg.run_tag,
                                                    model_variant=model_variant)
                binary_artifact_data = val_metrics.pop('_artifact_data', None)
            if train_cfg.task == 'stage':
                val_metrics, val_tracker = validate(model, val_loader, criterion, device, writer, epoch,
                                                    train_cfg.num_stages, list(train_cfg.class_names),
                                                    ordinal=train_cfg.ordinal, fold_save=fold_dir,
                                                    every_save_time=train_cfg.every_save_time,
                                                    task=train_cfg.task,
                                                    run_tag=train_cfg.run_tag,
                                                    model_variant=model_variant)
                metrics = val_tracker.compute(fold_id,epoch)
                val_metrics['accuracy'] = metrics['accuracy']
                save_metrics = {'epoch': epoch, 'run_tag': train_cfg.run_tag, 'model_variant': model_variant, **metrics}
                stage_metrics_path = os.path.join(fold_dir, 'stage_metrics.csv')
                pd.DataFrame([save_metrics]).to_csv(
                    stage_metrics_path,
                    mode='a',
                    index=False,
                    header=not os.path.exists(stage_metrics_path),
                )

            if feature_cloud_interval and feature_cloud_interval > 0 and epoch % feature_cloud_interval == 0:
                save_feature_clouds_on_demand(
                    model, train_loader, device, fold_dir, f"epoch_{epoch:03d}",
                    target="z", task=train_cfg.task, save_3d=train_cfg.feature_cloud_3d,
                    class_names=list(train_cfg.class_names),
                )

            if epoch % train_cfg.every_save_time == 0:
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{ckpt_stem}_epoch_{epoch}.pth"))
                history.append({
                    'epoch': epoch,
                    'run_tag': train_cfg.run_tag,
                    'model_variant': model_variant,
                    **train_metrics,
                    **val_metrics,
                })

            if train_cfg.task == 'stage':
                improved = (
                    val_metrics['accuracy'] > best_val_accuracy
                    or (
                        np.isclose(val_metrics['accuracy'], best_val_accuracy)
                        and val_metrics['val_loss'] < best_val_loss
                    )
                )
            else:
                improved = val_metrics['val_loss'] < best_val_loss

            save_validation_artifacts = (
                train_cfg.plot_interval > 0
                and epoch % train_cfg.plot_interval == 0
            ) or (train_cfg.plot_on_best and improved)
            if save_validation_artifacts:
                if train_cfg.task == 'bin' and binary_artifact_data is not None:
                    save_binary_validation_artifacts(
                        *binary_artifact_data,
                        fold_save=fold_dir,
                        epoch=epoch,
                    )
                elif train_cfg.task == 'stage' and val_tracker is not None:
                    return_stage_metrics(val_tracker, epoch)

            if improved:
                best_val_loss = val_metrics['val_loss']
                if train_cfg.task == 'stage':
                    best_val_accuracy = val_metrics['accuracy']
                counter = 0
                torch.save({
                    'epoch':              epoch,
                    'model_state_dict':   model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_metrics':        val_metrics,
                    'ablation_cfg':       vars(ablation_cfg),
                    'train_cfg':          vars(train_cfg),
                    'ssl_load_info':      ssl_load_info,
                    'model_variant':      model_variant,
                }, ckpt_path)
                if train_cfg.save_best_feature_cloud:
                    save_feature_clouds_on_demand(
                        model, train_loader, device, fold_dir, "best",
                        target="z", task=train_cfg.task, save_3d=train_cfg.feature_cloud_3d,
                        class_names=list(train_cfg.class_names),
                    )
            else:
                counter += 1
                if counter >= train_cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        hist_path = os.path.join(fold_dir, f'{ckpt_stem}_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        writer.close()
        completed.append({"fold": fold_id, "checkpoint": ckpt_path})

    return completed

def main():
    parser = argparse.ArgumentParser(description='SACON Training')
    parser.add_argument('--ablation', type=str, default='full',
                        help='消融配置名称（single）或 all（全部运行）')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=2)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes; use 0 for synchronous loading')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='batches prefetched by each worker when num_workers > 0')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='pin host tensors before CUDA transfer')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                        help='keep DataLoader workers alive between epochs')
    parser.add_argument('--lr',         type=float, default=8.96e-5)
    parser.add_argument("--num_stages",type=int,default=3)
    parser.add_argument('--task',       type=str,   default='stage', choices=['bin', 'stage'])
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--save_dir',   type=str,   default='./checkpoints')
    parser.add_argument('--ssl_ckpt', type=str, default=r"D:\projects\PHD2\ONFH_SACON\checkpoints\ssl_fastmri_clinical\fastmri_clinical_encoder_ssl_best.pth",
                        help='optional external SSL checkpoint used for both T1 and T2 encoders')
    parser.add_argument('--ssl_t1_ckpt', type=str, default=None,
                        help='optional SSL checkpoint for SACON t1_encoder')
    parser.add_argument('--ssl_t2_ckpt', type=str, default=None,
                        help='optional SSL checkpoint for SACON t2_encoder')
    parser.add_argument('--siemens_finetune', type=str, default="", metavar='CHECKPOINT',
                        help='optional full SIEMENS supervised checkpoint used to initialize all model weights')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0,
                        help='freeze T1/T2 encoders for the first N fine-tuning epochs')
    parser.add_argument('--run_tag', type=str, default='ssl-uih',
                        help='optional suffix for checkpoint names, e.g. ssl')
    parser.add_argument('--group_by_base_id', action='store_true',default=False,
                        help='group offline augmented samples by base patient ID during 5-fold CV')
    parser.add_argument('--repeated_holdout_7_3', action='store_true', default=False,
                        help='run 5 repeated stratified base-patient-ID holdouts with a 70/30 train/validation ratio')
    parser.add_argument(
        '--real_only_test',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='数据层面重复留出：全部合成数据进入训练集，测试集仅包含真实数据',
    )
    parser.add_argument(
        '--real_test_ratio',
        type=parse_real_test_ratio,
        default=0.3,
        help='每次测试集占全部真实样本的比例，必须满足 0 < ratio < 1',
    )
    parser.add_argument('--online_train_augment', action='store_true',default=False,
                        help='enable virtual online train augmentation: late +5 copies, non-late +1 copy by default')
    parser.add_argument('--late_aug_copies', type=int, default=5,
                        help='number of online augmented copies for late cases')
    parser.add_argument('--nonlate_aug_copies', type=int, default=0,
                        help='number of online augmented copies for non-late cases')
    parser.add_argument('--late_aug_max_retries', type=int, default=30,
                        help='maximum QC retries for each online augmented copy')
    parser.add_argument('--feature_cloud_interval', type=int, default=5,
                        help='save feature cloud every N epochs; default follows every_save_time, 0 disables')
    parser.add_argument('--feature_cloud_3d', action=argparse.BooleanOptionalAction, default=True,
                        help='save 3D feature cloud PNG')
    parser.add_argument('--save_best_feature_cloud', action=argparse.BooleanOptionalAction, default=False,
                        help='run an extra full training-set pass when a new best checkpoint is found')
    parser.add_argument('--grad_flow_interval', type=int, default=0,
                        help='save gradient interaction every N epochs; default follows every_save_time, 0 disables')
    parser.add_argument('--grad_flow_samples', type=int, default=1,
                        help='number of training batches used for gradient flow figures at each logging epoch')
    parser.add_argument('--tensorboard_image_interval', type=int, default=0,
                        help='write feature-map images every N training steps; 0 disables')
    parser.add_argument('--progress_update_interval', type=int, default=20,
                        help='update CUDA-backed tqdm loss values every N batches; 0 disables')
    parser.add_argument('--enable_cam_hooks', action=argparse.BooleanOptionalAction, default=False,
                        help='register feature hooks for CAM diagnostics')
    parser.add_argument('--plot_interval', type=int, default=0,
                        help='save validation plots every N epochs; 0 disables periodic plots')
    parser.add_argument('--plot_on_best', action=argparse.BooleanOptionalAction, default=True,
                        help='save validation plots whenever a new best checkpoint is found')
    args = parser.parse_args()

    if args.num_workers < 0:
        parser.error('--num_workers must be >= 0')
    if args.num_workers > 0 and args.prefetch_factor < 1:
        parser.error('--prefetch_factor must be >= 1 when workers are enabled')
    for name in (
        'feature_cloud_interval',
        'grad_flow_interval',
        'tensorboard_image_interval',
        'progress_update_interval',
        'plot_interval',
    ):
        if getattr(args, name) < 0:
            parser.error(f'--{name} must be >= 0')

    ssl_t1_ckpt = args.ssl_t1_ckpt
    ssl_t2_ckpt = args.ssl_t2_ckpt
    if args.ssl_ckpt:
        ssl_t1_ckpt = ssl_t1_ckpt or args.ssl_ckpt
        ssl_t2_ckpt = ssl_t2_ckpt or args.ssl_ckpt

    train_cfg = TrainConfig(
        data_root = args.data_root,
        num_epochs=args.epochs,
        num_stages = args.num_stages,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        lr=args.lr,
        task=args.task,
        device=args.device,
        save_dir=args.save_dir,
        ssl_t1_ckpt=ssl_t1_ckpt,
        ssl_t2_ckpt=ssl_t2_ckpt,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        run_tag=args.run_tag,
        group_by_base_id=args.group_by_base_id,
        repeated_holdout_7_3=args.repeated_holdout_7_3,
        real_only_test=args.real_only_test,
        real_test_ratio=args.real_test_ratio,
        online_train_augment=args.online_train_augment,
        late_aug_copies=args.late_aug_copies,
        nonlate_aug_copies=args.nonlate_aug_copies,
        late_aug_max_retries=args.late_aug_max_retries,
        feature_cloud_interval=args.feature_cloud_interval,
        feature_cloud_3d=args.feature_cloud_3d,
        save_best_feature_cloud=args.save_best_feature_cloud,
        grad_flow_interval=args.grad_flow_interval,
        grad_flow_samples=args.grad_flow_samples,
        tensorboard_image_interval=args.tensorboard_image_interval,
        progress_update_interval=args.progress_update_interval,
        enable_cam_hooks=args.enable_cam_hooks,
        plot_interval=args.plot_interval,
        plot_on_best=args.plot_on_best,
    )


    run_experiment(args.ablation, train_cfg, siemens_finetune_ckpt=args.siemens_finetune)

if __name__ == '__main__':
    main()
