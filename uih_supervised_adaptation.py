
from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from configs.config import AblationConfig
from data.dataset import ONFHDataset
from evaluate_external_protocol import (
    infer_predictions,
    load_model_from_checkpoint,
)
from losses.losses import SACANLoss
from models.sacon import SACON
from utils.ssl_weights import load_ssl_encoder_weights, set_sacon_encoder_trainable


PROJECT_ROOT = Path(__file__).resolve().parent
STRATEGIES = ("zero_shot", "siemens_finetune", "uih_ssl_only")
TASKS = ("bin", "stage")
PRIMARY_METRIC = {"bin": "auroc", "stage": "qwk"}
COLORS = {
    "zero_shot": "#7A7A7A",
    "siemens_finetune": "#0072B2",
    "uih_ssl_only": "#E69F00",
}


@dataclass(frozen=True)
class AdaptationConfig:

    uih_data_root: Path = Path(
        r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH\uih_onfh_data_resampled.json"
    )
    bin_source_checkpoint: Path = Path(
        r"E:\BackupToDriveF\【重要】文章-汇总\【Article】SACON_-Symmetry-Aware Contralateral-difference Analysis-\SACON-原始结果\合成数据_50_7_ssl_SACON_二分类任务\bin_fold_1\bin_full_ssl_best.pth"
    )
    stage_source_checkpoint: Path = Path(
        r"E:\BackupToDriveF\【重要】文章-汇总\【Article】SACON_-Symmetry-Aware Contralateral-difference Analysis-\SACON-原始结果\合成数据_50_7_ssl_SACON_分期任务\stage_fold_4\stage_full_ssl_best.pth"
    )
    ssl_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "ssl_fastmri_clinical" / "fastmri_clinical_encoder_ssl_best.pth"
    tasks: Tuple[str, ...] = ("bin", "stage")
    output_root: Path = PROJECT_ROOT / "outputs" / "uih_adaptation"

    outer_folds: int = 5
    inner_val_ratio: float = 0.20
    split_seed: int = 42
    training_seeds: Tuple[int, ...] = (42, 2026, 3407)

    max_epochs: int = 50
    patience: int = 10
    freeze_encoder_epochs: int = 5
    encoder_lr: float = 1e-5
    finetune_other_lr: float = 5e-5
    ssl_only_other_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    num_workers: int = 0
    device: str = "cuda"
    volume_size: Tuple[int, int, int] = (19, 314, 224)
    threshold: float = 0.5
    n_bootstrap: int = 2000
    reuse_existing: bool = True
    preflight_only: bool = False


DIRECT_CONFIG = AdaptationConfig()


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(payload)), handle, ensure_ascii=False, indent=2)


def load_manifest(data_root: Path) -> OrderedDict:
    with data_root.open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=OrderedDict)
    required = {
        "patient_id",
        "left_t1_path",
        "left_t2_path",
        "right_t1_path",
        "right_t2_path",
        "label_left_bin",
        "label_right_bin",
        "label_left_ELS",
        "label_right_ELS",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"UIH JSON 缺少必要字段：{sorted(missing)}")
    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"UIH JSON 各字段长度不一致：{lengths}")
    patient_ids = [str(value) for value in data["patient_id"]]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("UIH JSON 存在重复 patient_id，必须先按患者去重后再划分。")
    return data


def patient_stratify_labels(data: Mapping[str, Sequence[object]], task: str) -> np.ndarray:
    if task == "bin":
        return np.asarray(data["label_left_bin"], dtype=int) + np.asarray(data["label_right_bin"], dtype=int)
    return np.maximum(
        np.asarray(data["label_left_ELS"], dtype=int),
        np.asarray(data["label_right_ELS"], dtype=int),
    )


def build_nested_splits(
    data: Mapping[str, Sequence[object]],
    task: str,
    outer_folds: int,
    inner_val_ratio: float,
    split_seed: int,
) -> List[Dict[str, np.ndarray]]:
    indices = np.arange(len(data["patient_id"]), dtype=np.int64)
    labels = patient_stratify_labels(data, task)
    counts = pd.Series(labels).value_counts().to_dict()
    if min(counts.values()) < outer_folds:
        raise ValueError(f"{task} 分层中最小类别少于 {outer_folds} 例：{counts}")

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=split_seed)
    splits: List[Dict[str, np.ndarray]] = []
    for fold, (outer_train, outer_test) in enumerate(outer.split(indices, labels)):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=inner_val_ratio,
            random_state=split_seed + fold + 1,
        )
        inner_train_position, inner_val_position = next(
            inner.split(outer_train, labels[outer_train])
        )
        train_ids = outer_train[inner_train_position]
        val_ids = outer_train[inner_val_position]
        test_ids = outer_test
        if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
            raise RuntimeError(f"{task} fold {fold} 检测到训练/验证/测试患者泄漏。")
        splits.append({"fold": np.asarray([fold]), "train": train_ids, "val": val_ids, "test": test_ids})

    test_union = np.concatenate([item["test"] for item in splits])
    if sorted(test_union.tolist()) != indices.tolist():
        raise RuntimeError(f"{task} 外层测试折未做到每名患者恰好覆盖一次。")
    return splits


def split_manifest(
    data: Mapping[str, Sequence[object]],
    task: str,
    splits: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    labels = patient_stratify_labels(data, task)
    patient_ids = [str(value) for value in data["patient_id"]]
    rows = []
    for item in splits:
        fold = int(item["fold"][0])
        row: Dict[str, object] = {"fold": fold}
        for part in ("train", "val", "test"):
            ids = item[part]
            row[f"n_{part}"] = int(len(ids))
            row[f"{part}_label_counts"] = pd.Series(labels[ids]).value_counts().sort_index().to_dict()
            row[f"{part}_patient_ids"] = [patient_ids[index] for index in ids]
        rows.append(row)
    return {"task": task, "stratification": "positive_hip_count" if task == "bin" else "maximum_bilateral_stage", "folds": rows}


def make_loader(
    config: AdaptationConfig,
    ids: Sequence[int],
    augment: bool,
    shuffle: bool,
    seed: int,
) -> Tuple[ONFHDataset, data_utils.DataLoader]:
    dataset = ONFHDataset(
        data_root=str(config.uih_data_root),
        ids=list(map(int, ids)),
        volume_size=config.volume_size,
        augment=augment,
        online_train_augment=False,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = data_utils.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def source_checkpoint_for(config: AdaptationConfig, task: str) -> Path:
    return config.bin_source_checkpoint if task == "bin" else config.stage_source_checkpoint


def validate_source_checkpoint(path: Path, task: str, config: AdaptationConfig) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    train_cfg = checkpoint.get("train_cfg", {})
    checkpoint_task = train_cfg.get("task")
    if checkpoint_task and checkpoint_task != task:
        raise ValueError(f"checkpoint 任务为 {checkpoint_task}，但当前请求为 {task}：{path}")
    saved_size = tuple(train_cfg.get("volume_size", config.volume_size))
    if saved_size != tuple(config.volume_size):
        raise ValueError(f"输入尺寸不一致：checkpoint={saved_size}，当前={config.volume_size}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint 未包含 model_state_dict：{path}")
    return checkpoint


def build_ssl_only_model(
    source_metadata: Mapping[str, object],
    config: AdaptationConfig,
    task: str,
    device: torch.device,
) -> Tuple[SACON, Dict[str, object]]:
    train_cfg = dict(source_metadata.get("train_cfg", {}))
    ablation_cfg = AblationConfig(**dict(source_metadata.get("ablation_cfg", {})))
    model = SACON(
        ablation_cfg,
        base_channels=int(train_cfg.get("base_channels", 32)),
        num_stages=int(train_cfg.get("num_stages", 3)),
        ordinal=bool(train_cfg.get("ordinal", True)),
        task=task,
    ).to(device)
    load_info = load_ssl_encoder_weights(
        model,
        t1_ckpt=str(config.ssl_checkpoint),
        t2_ckpt=str(config.ssl_checkpoint),
        device=device,
        strict=False,
    )
    return model, load_info


def build_criterion(
    model: SACON,
    source_metadata: Mapping[str, object],
    train_dataset: ONFHDataset,
    task: str,
    device: torch.device,
) -> SACANLoss:
    train_cfg = dict(source_metadata.get("train_cfg", {}))
    ablation_cfg = model.config
    criterion = SACANLoss(
        ablation_cfg,
        alpha_bin=float(train_cfg.get("alpha_bin", 1.2)),
        alpha_stage=float(train_cfg.get("alpha_stage", 0.82)),
        alpha_reg=float(train_cfg.get("alpha_reg", 0.25 if task == "bin" else 0.18)),
        alpha_decouple=float(train_cfg.get("alpha_decouple", 0.01 if task == "bin" else 0.07)),
        alpha_conc=float(train_cfg.get("alpha_conc", 0.005)),
        num_stages=int(train_cfg.get("num_stages", 3)),
        ordinal=bool(train_cfg.get("ordinal", True)),
        stage_class_weights=train_dataset.return_stage_class_weights(),
        bin_class_weights=train_dataset.return_bin_class_weights(),
        task=task,
    )
    return criterion.to(device)


def labels_from_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "left_binary": batch["left_binary"].to(device),
        "right_binary": batch["right_binary"].to(device),
        "left_stage": batch["label_left_ELS"].to(device),
        "right_stage": batch["label_right_ELS"].to(device),
    }


def forward_loss(
    model: SACON,
    criterion: SACANLoss,
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    outputs = model(
        batch["left_t1"].to(device),
        batch["left_t2"].to(device),
        batch["right_t1"].to(device),
        batch["right_t2"].to(device),
    )
    loss, _ = criterion(outputs, labels_from_batch(batch, device))
    return loss


def train_epoch(
    model: SACON,
    criterion: SACANLoss,
    loader: data_utils.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    description: str,
) -> float:
    model.train()
    losses: List[float] = []
    for batch in tqdm(loader, desc=description, leave=False):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            loss = forward_loss(model, criterion, batch, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def validation_loss(
    model: SACON,
    criterion: SACANLoss,
    loader: data_utils.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        with autocast(enabled=device.type == "cuda"):
            loss = forward_loss(model, criterion, batch, device)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def optimizer_for(
    model: SACON,
    config: AdaptationConfig,
    strategy: str,
) -> torch.optim.Optimizer:
    encoder_ids = {
        id(parameter)
        for module in (model.t1_encoder, model.t2_encoder)
        for parameter in module.parameters()
    }
    encoder_parameters = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    other_lr = config.finetune_other_lr if strategy == "siemens_finetune" else config.ssl_only_other_lr
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": config.encoder_lr},
            {"params": other_parameters, "lr": other_lr},
        ],
        weight_decay=config.weight_decay,
    )


def train_one_model(
    config: AdaptationConfig,
    task: str,
    strategy: str,
    fold: int,
    seed: int,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    device: torch.device,
    run_dir: Path,
) -> Tuple[SACON, pd.DataFrame, Path]:
    set_reproducible(seed)
    source_path = source_checkpoint_for(config, task)
    source_metadata = validate_source_checkpoint(source_path, task, config)
    train_dataset, train_loader = make_loader(config, train_ids, augment=True, shuffle=True, seed=seed)
    _, val_loader = make_loader(config, val_ids, augment=False, shuffle=False, seed=seed)

    if strategy == "siemens_finetune":
        model, _ = load_model_from_checkpoint(str(source_path), task, device)
        initialization = {"type": "full_siemens_supervised", "path": str(source_path)}
    elif strategy == "uih_ssl_only":
        model, ssl_info = build_ssl_only_model(source_metadata, config, task, device)
        initialization = {"type": "ssl_encoders_only", "path": str(config.ssl_checkpoint), "load_info": ssl_info}
    else:
        raise ValueError(f"未知训练策略：{strategy}")

    criterion = build_criterion(model, source_metadata, train_dataset, task, device)
    optimizer = optimizer_for(model, config, strategy)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    scaler = GradScaler(enabled=device.type == "cuda")
    checkpoint_path = run_dir / "best.pth"
    history_rows: List[Dict[str, object]] = []
    best_loss = float("inf")
    bad_epochs = 0

    if config.freeze_encoder_epochs > 0:
        set_sacon_encoder_trainable(model, False)

    for epoch in range(1, config.max_epochs + 1):
        if config.freeze_encoder_epochs > 0 and epoch == config.freeze_encoder_epochs + 1:
            set_sacon_encoder_trainable(model, True)
        train_loss = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            description=f"{task}/{strategy}/fold{fold}/seed{seed}/epoch{epoch}",
        )
        val_loss = validation_loss(model, criterion, val_loader, device)
        history_rows.append(
            {
                "task": task,
                "strategy": strategy,
                "fold": fold,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "other_lr": optimizer.param_groups[1]["lr"],
            }
        )

        if np.isfinite(val_loss) and val_loss < best_loss - 1e-6:
            best_loss = val_loss
            bad_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": {"val_loss": val_loss},
                    "ablation_cfg": asdict(model.config),
                    "train_cfg": {
                        **dict(source_metadata.get("train_cfg", {})),
                        "task": task,
                        "data_root": str(config.uih_data_root),
                        "volume_size": tuple(config.volume_size),
                    },
                    "adaptation": {
                        "strategy": strategy,
                        "fold": fold,
                        "seed": seed,
                        "initialization": initialization,
                        "n_train_patients": len(train_ids),
                        "n_val_patients": len(val_ids),
                    },
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        scheduler.step()
        if bad_epochs >= config.patience:
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"训练未生成有效checkpoint：{run_dir}")
    best_model, _ = load_model_from_checkpoint(str(checkpoint_path), task, device)
    return best_model, history, checkpoint_path


def add_run_columns(
    predictions: pd.DataFrame,
    task: str,
    strategy: str,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    result = predictions.copy()
    result["cohort"] = "UIH"
    result["task"] = task
    result["strategy"] = strategy
    result["outer_fold"] = fold
    result["seed"] = seed
    return result


def predict_model(
    model: SACON,
    config: AdaptationConfig,
    task: str,
    ids: Sequence[int],
    device: torch.device,
    checkpoint_name: str,
) -> pd.DataFrame:
    dataset, loader = make_loader(config, ids, augment=False, shuffle=False, seed=config.split_seed)
    return infer_predictions(
        model=model,
        loader=loader,
        dataset=dataset,
        task=task,
        checkpoint_name=checkpoint_name,
        device=device,
        num_classes=3,
        threshold=config.threshold,
    )


def aggregate_seed_predictions(raw: pd.DataFrame, task: str, threshold: float) -> pd.DataFrame:
    keys = ["cohort", "task", "strategy", "outer_fold", "patient_id", "source_index", "side", "label"]
    rows = []
    for key_values, group in raw.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key_values))
        row["n_seeds"] = int(group["seed"].nunique())
        if task == "bin":
            probability = float(group["prob_positive"].mean())
            prediction = int(probability >= threshold)
            row.update(
                {
                    "prob_positive": probability,
                    "prediction": prediction,
                    "confidence": max(probability, 1.0 - probability),
                    "is_error": prediction != int(row["label"]),
                }
            )
        else:
            probabilities = np.asarray(
                [float(group[f"prob_class_{index}"].mean()) for index in range(3)]
            )
            probabilities /= np.clip(probabilities.sum(), 1e-12, None)
            prediction = int(np.argmax(probabilities))
            for index, value in enumerate(probabilities):
                row[f"prob_class_{index}"] = float(value)
            row.update(
                {
                    "prediction": prediction,
                    "confidence": float(probabilities.max()),
                    "abs_error": abs(prediction - int(row["label"])),
                    "is_error": prediction != int(row["label"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_oof_predictions(predictions: pd.DataFrame, n_patients: int, task: str) -> None:
    expected_rows = n_patients * 2
    for strategy in STRATEGIES:
        group = predictions[predictions["strategy"] == strategy]
        if len(group) != expected_rows:
            raise RuntimeError(f"{task}/{strategy} 应有 {expected_rows} 髋预测，当前为 {len(group)}。")
        if group["patient_id"].nunique() != n_patients:
            raise RuntimeError(f"{task}/{strategy} 未覆盖全部 {n_patients} 名患者。")
        if group.duplicated(["patient_id", "side"]).any():
            raise RuntimeError(f"{task}/{strategy} 存在重复患者-侧别 OOF 预测。")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        mask = indices == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def metric_values(frame: pd.DataFrame, task: str, threshold: float) -> Dict[str, float]:
    y_true = frame["label"].to_numpy(int)
    if task == "bin":
        y_prob = frame["prob_positive"].to_numpy(float)
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "auroc": float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else np.nan,
            "auprc": float(average_precision_score(y_true, y_prob)) if np.unique(y_true).size > 1 else np.nan,
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "ece": expected_calibration_error(y_true, y_prob),
        }
    y_pred = frame["prediction"].to_numpy(int)
    absolute_error = np.abs(y_pred - y_true)
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2], weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within_one_stage_accuracy": float((absolute_error <= 1).mean()),
        "severe_cross_stage_error_rate": float((absolute_error > 1).mean()),
    }


def resample_patients(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    patients = frame["patient_id"].astype(str).drop_duplicates().to_numpy()
    sampled = rng.choice(patients, size=len(patients), replace=True)
    groups = {patient: group for patient, group in frame.groupby(frame["patient_id"].astype(str), sort=False)}
    return pd.concat([groups[patient] for patient in sampled], ignore_index=True)


def summarize_metrics(
    predictions: pd.DataFrame,
    task: str,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for strategy, group in predictions.groupby("strategy", sort=False):
        point = metric_values(group, task, threshold)
        rng = np.random.default_rng(seed)
        bootstrap = {metric: [] for metric in point}
        for _ in range(n_bootstrap):
            values = metric_values(resample_patients(group, rng), task, threshold)
            for metric, value in values.items():
                if np.isfinite(value):
                    bootstrap[metric].append(value)
        for metric, estimate in point.items():
            values = np.asarray(bootstrap[metric], dtype=float)
            rows.append(
                {
                    "task": task,
                    "strategy": strategy,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.percentile(values, 2.5)) if values.size else np.nan,
                    "ci_high": float(np.percentile(values, 97.5)) if values.size else np.nan,
                    "n_patients": group["patient_id"].nunique(),
                    "n_hips": len(group),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(values[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_metric_differences(
    predictions: pd.DataFrame,
    task: str,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    contrasts = (
        ("siemens_finetune", "zero_shot"),
        ("siemens_finetune", "uih_ssl_only"),
        ("uih_ssl_only", "zero_shot"),
    )
    frames = {strategy: group.copy() for strategy, group in predictions.groupby("strategy")}
    reference_patients = frames["zero_shot"]["patient_id"].astype(str).drop_duplicates().to_numpy()
    grouped = {
        strategy: {patient: group for patient, group in frame.groupby(frame["patient_id"].astype(str), sort=False)}
        for strategy, frame in frames.items()
    }
    rows = []
    for left, right in contrasts:
        left_point = metric_values(frames[left], task, threshold)
        right_point = metric_values(frames[right], task, threshold)
        bootstrap = {metric: [] for metric in left_point}
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            sampled = rng.choice(reference_patients, size=len(reference_patients), replace=True)
            left_sample = pd.concat([grouped[left][patient] for patient in sampled], ignore_index=True)
            right_sample = pd.concat([grouped[right][patient] for patient in sampled], ignore_index=True)
            left_values = metric_values(left_sample, task, threshold)
            right_values = metric_values(right_sample, task, threshold)
            for metric in bootstrap:
                difference = left_values[metric] - right_values[metric]
                if np.isfinite(difference):
                    bootstrap[metric].append(difference)
        for metric, left_estimate in left_point.items():
            difference = left_estimate - right_point[metric]
            values = np.asarray(bootstrap[metric], dtype=float)
            p_value = (
                min(1.0, 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean())))
                if values.size else np.nan
            )
            rows.append(
                {
                    "task": task,
                    "comparison": f"{left} vs {right}",
                    "metric": metric,
                    "difference": difference,
                    "ci_low": float(np.percentile(values, 2.5)) if values.size else np.nan,
                    "ci_high": float(np.percentile(values, 97.5)) if values.size else np.nan,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for _, indices in result.groupby("metric").groups.items():
        result.loc[indices, "p_holm"] = holm_adjust(result.loc[indices, "p_value"])
    return result


def summarize_seed_variability(raw_predictions: pd.DataFrame, task: str, threshold: float) -> pd.DataFrame:
    rows = []
    for (strategy, seed), group in raw_predictions.groupby(["strategy", "seed"], sort=False):
        values = metric_values(group, task, threshold)
        rows.extend(
            {"task": task, "strategy": strategy, "seed": seed, "metric": metric, "value": value}
            for metric, value in values.items()
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_primary_metrics(metrics: pd.DataFrame, output_root: Path) -> None:
    set_publication_style()
    tasks = [task for task in TASKS if task in set(metrics["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.45 * len(tasks), 2.35), squeeze=False)
    for panel, task in enumerate(tasks):
        ax = axes[0, panel]
        metric = PRIMARY_METRIC[task]
        subset = metrics[(metrics["task"] == task) & (metrics["metric"] == metric)].copy()
        subset["strategy"] = pd.Categorical(subset["strategy"], STRATEGIES, ordered=True)
        subset = subset.sort_values("strategy")
        for index, row in enumerate(subset.itertuples(index=False)):
            ax.errorbar(
                row.estimate,
                index,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt="o",
                color=COLORS[str(row.strategy)],
                capsize=2,
            )
        ax.set_yticks(range(len(subset)), [str(value) for value in subset["strategy"]])
        ax.set_xlabel(("AUROC" if task == "bin" else "Quadratic weighted kappa") + " (95% CI)")
        ax.set_title("Binary classification" if task == "bin" else "Stage classification")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_root / "figures" / "fig_strategy_primary_metrics")


def plot_primary_differences(differences: pd.DataFrame, output_root: Path) -> None:
    set_publication_style()
    selected = differences[
        differences.apply(lambda row: row["metric"] == PRIMARY_METRIC.get(row["task"]), axis=1)
    ].copy()
    selected["label"] = selected["task"].astype(str) + " | " + selected["comparison"].astype(str)
    y = np.arange(len(selected))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, max(2.2, 0.38 * len(selected) + 0.7)))
    for position, row in zip(y, selected.itertuples(index=False)):
        color = "#D55E00" if row.p_holm < 0.05 else "#0072B2"
        ax.errorbar(
            row.difference,
            position,
            xerr=[[row.difference - row.ci_low], [row.ci_high - row.difference]],
            fmt="o",
            color=color,
            capsize=2,
        )
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, selected["label"])
    ax.set_xlabel("Difference in primary metric (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_root / "figures" / "fig_strategy_primary_differences")


def plot_task_diagnostics(predictions: pd.DataFrame, task: str, output_root: Path) -> None:
    set_publication_style()
    figure_dir = output_root / "figures"
    if task == "bin":
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
        prevalence = predictions["label"].mean()
        for strategy in STRATEGIES:
            group = predictions[predictions["strategy"] == strategy]
            y_true = group["label"].to_numpy(int)
            y_prob = group["prob_positive"].to_numpy(float)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            observed, predicted = calibration_curve(
                y_true,
                y_prob,
                n_bins=6,
                strategy="quantile",
            )
            axes[0].plot(fpr, tpr, color=COLORS[strategy], label=f"{strategy} ({roc_auc_score(y_true, y_prob):.3f})")
            axes[1].plot(recall, precision, color=COLORS[strategy], label=f"{strategy} ({average_precision_score(y_true, y_prob):.3f})")
            axes[2].plot(predicted, observed, marker="o", markersize=2.5, color=COLORS[strategy], label=strategy)
        axes[0].plot([0, 1], [0, 1], color="#777777", linestyle=":")
        axes[1].axhline(prevalence, color="#777777", linestyle=":")
        axes[2].plot([0, 1], [0, 1], color="#777777", linestyle=":")
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
        axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        axes[2].set(xlabel="Mean predicted probability", ylabel="Observed proportion", xlim=(0, 1), ylim=(0, 1.02))
        for ax in axes:
            ax.legend(fontsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_figure(fig, figure_dir / "fig_binary_strategy_curves")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), squeeze=False)
    image = None
    for index, strategy in enumerate(STRATEGIES):
        group = predictions[predictions["strategy"] == strategy]
        matrix = confusion_matrix(group["label"], group["prediction"], labels=[0, 1, 2]).astype(float)
        matrix /= np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
        ax = axes[0, index]
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for column in range(3):
                ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if matrix[row, column] > 0.55 else "black")
        ax.set_xticks(range(3), ["Normal", "Early", "Late"], rotation=35, ha="right")
        ax.set_yticks(range(3), ["Normal", "Early", "Late"] if index == 0 else [])
        ax.set_xlabel("Predicted")
        if index == 0:
            ax.set_ylabel("True")
        ax.set_title(strategy)
        ax.tick_params(direction="out", length=0)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Row proportion")
    fig.subplots_adjust(wspace=0.28)
    save_figure(fig, figure_dir / "fig_stage_strategy_confusions")


def run_task(
    config: AdaptationConfig,
    task: str,
    data: Mapping[str, Sequence[object]],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_dir = config.output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    splits = build_nested_splits(data, task, config.outer_folds, config.inner_val_ratio, config.split_seed)
    save_json(split_manifest(data, task, splits), task_dir / "split_manifest.json")
    source_path = source_checkpoint_for(config, task)
    validate_source_checkpoint(source_path, task, config)
    all_raw_predictions: List[pd.DataFrame] = []

    for split in splits:
        fold = int(split["fold"][0])
        test_ids = split["test"]

        zero_dir = task_dir / "zero_shot" / f"fold_{fold}"
        zero_path = zero_dir / "predictions.csv"
        if config.reuse_existing and zero_path.is_file():
            zero_predictions = pd.read_csv(zero_path)
        else:
            zero_dir.mkdir(parents=True, exist_ok=True)
            zero_model, _ = load_model_from_checkpoint(str(source_path), task, device)
            zero_predictions = predict_model(zero_model, config, task, test_ids, device, source_path.stem)
            zero_predictions = add_run_columns(zero_predictions, task, "zero_shot", fold, -1)
            zero_predictions.to_csv(zero_path, index=False, encoding="utf-8-sig")
            del zero_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_raw_predictions.append(zero_predictions)

        for strategy in ("siemens_finetune", "uih_ssl_only"):
            for seed in config.training_seeds:
                run_dir = task_dir / strategy / f"fold_{fold}" / f"seed_{seed}"
                prediction_path = run_dir / "predictions.csv"
                if config.reuse_existing and prediction_path.is_file():
                    predictions = pd.read_csv(prediction_path)
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = run_dir / "best.pth"
                    if config.reuse_existing and checkpoint_path.is_file():
                        model, _ = load_model_from_checkpoint(str(checkpoint_path), task, device)
                    else:
                        model, _, checkpoint_path = train_one_model(
                            config=config,
                            task=task,
                            strategy=strategy,
                            fold=fold,
                            seed=seed,
                            train_ids=split["train"],
                            val_ids=split["val"],
                            device=device,
                            run_dir=run_dir,
                        )
                    predictions = predict_model(model, config, task, test_ids, device, checkpoint_path.stem)
                    predictions = add_run_columns(predictions, task, strategy, fold, seed)
                    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                all_raw_predictions.append(predictions)

    raw = pd.concat(all_raw_predictions, ignore_index=True)
    raw.to_csv(task_dir / "predictions_by_seed.csv", index=False, encoding="utf-8-sig")
    aggregated_parts = []
    for strategy, group in raw.groupby("strategy", sort=False):
        aggregated_parts.append(aggregate_seed_predictions(group, task, config.threshold))
    oof = pd.concat(aggregated_parts, ignore_index=True)
    validate_oof_predictions(oof, len(data["patient_id"]), task)
    oof.to_csv(task_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = summarize_metrics(oof, task, config.threshold, config.n_bootstrap, config.split_seed + 100)
    metrics.to_csv(task_dir / "strategy_metrics.csv", index=False, encoding="utf-8-sig")
    differences = paired_metric_differences(oof, task, config.threshold, config.n_bootstrap, config.split_seed + 200)
    differences.to_csv(task_dir / "paired_strategy_differences.csv", index=False, encoding="utf-8-sig")
    seed_metrics = summarize_seed_variability(raw, task, config.threshold)
    seed_metrics.to_csv(task_dir / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    plot_task_diagnostics(oof, task, config.output_root)
    return oof, metrics, differences


def preflight(config: AdaptationConfig) -> Tuple[OrderedDict, Dict[str, object]]:
    errors = []
    if not config.uih_data_root.is_file():
        errors.append(f"UIH JSON 不存在：{config.uih_data_root}")
    if not config.ssl_checkpoint.is_file():
        errors.append(f"SSL checkpoint 不存在：{config.ssl_checkpoint}")
    for task in config.tasks:
        if task not in TASKS:
            errors.append(f"未知任务：{task}")
            continue
        path = source_checkpoint_for(config, task)
        if not path.is_file():
            errors.append(f"{task} SIEMENS checkpoint 不存在：{path}")
    if not 0 < config.inner_val_ratio < 0.5:
        errors.append("inner_val_ratio 必须位于 0 和 0.5 之间。")
    if config.outer_folds < 2:
        errors.append("outer_folds 至少为 2。")
    if not config.training_seeds:
        errors.append("training_seeds 不能为空。")
    if errors:
        raise FileNotFoundError("UIH迁移实验预检查未通过：\n" + "\n".join(f"  - {item}" for item in errors))

    data = load_manifest(config.uih_data_root)
    report: Dict[str, object] = {
        "n_patients": len(data["patient_id"]),
        "tasks": list(config.tasks),
        "device_requested": config.device,
        "cuda_available": torch.cuda.is_available(),
        "splits": {},
    }
    for task in config.tasks:
        validate_source_checkpoint(source_checkpoint_for(config, task), task, config)
        splits = build_nested_splits(data, task, config.outer_folds, config.inner_val_ratio, config.split_seed)
        report["splits"][task] = split_manifest(data, task, splits)
    return data, report


def run_experiment(config: AdaptationConfig) -> None:
    data, preflight_report = preflight(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    save_json(preflight_report, config.output_root / "preflight_report.json")
    save_json({"config": asdict(config)}, config.output_root / "experiment_config.json")
    print(f"[OK] 预检查通过：UIH患者数={len(data['patient_id'])}，任务={list(config.tasks)}")
    if config.preflight_only:
        print(f"[STOP] preflight_only=True，报告已保存至：{config.output_root}")
        return

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    all_oof, all_metrics, all_differences = [], [], []
    for task in config.tasks:
        print(f"[RUN] 开始任务：{task}，设备：{device}")
        oof, metrics, differences = run_task(config, task, data, device)
        all_oof.append(oof)
        all_metrics.append(metrics)
        all_differences.append(differences)

    combined_oof = pd.concat(all_oof, ignore_index=True)
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_differences = pd.concat(all_differences, ignore_index=True)
    combined_oof.to_csv(config.output_root / "all_oof_predictions.csv", index=False, encoding="utf-8-sig")
    combined_metrics.to_csv(config.output_root / "all_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    combined_differences.to_csv(config.output_root / "all_paired_differences.csv", index=False, encoding="utf-8-sig")
    plot_primary_metrics(combined_metrics, config.output_root)
    plot_primary_differences(combined_differences, config.output_root)
    save_json(
        {
            "status": "complete",
            "n_patients": len(data["patient_id"]),
            "strategies": list(STRATEGIES),
            "tasks": list(config.tasks),
            "interpretation_boundary": "UIH supervised target-domain adaptation; not independent zero-shot external validation",
        },
        config.output_root / "experiment_manifest.json",
    )
    print(f"[OK] UIH迁移评估完成：{config.output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SACON在UIH上的zero-shot、SIEMENS监督微调与SSL-only再训练嵌套五折评估。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uih_data_root", default=str(DIRECT_CONFIG.uih_data_root), help="UIH数据清单JSON路径。")
    parser.add_argument("--bin_source_checkpoint", default=str(DIRECT_CONFIG.bin_source_checkpoint), help="SIEMENS二分类完整监督checkpoint。")
    parser.add_argument("--stage_source_checkpoint", default=str(DIRECT_CONFIG.stage_source_checkpoint), help="SIEMENS分期完整监督checkpoint。")
    parser.add_argument("--ssl_checkpoint", default=str(DIRECT_CONFIG.ssl_checkpoint), help="UIH SSL-only对照使用的编码器预训练权重。")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(DIRECT_CONFIG.tasks), help="需要执行的任务，可选择bin、stage或同时选择。")
    parser.add_argument("--output_root", default=str(DIRECT_CONFIG.output_root), help="模型、预测、指标和图形输出根目录。")
    parser.add_argument("--outer_folds", type=int, default=DIRECT_CONFIG.outer_folds, help="患者级外层测试折数。")
    parser.add_argument("--inner_val_ratio", type=float, default=DIRECT_CONFIG.inner_val_ratio, help="外层训练数据中划作早停验证集的比例。")
    parser.add_argument("--split_seed", type=int, default=DIRECT_CONFIG.split_seed, help="患者五折划分随机种子。")
    parser.add_argument("--training_seeds", type=int, nargs="+", default=list(DIRECT_CONFIG.training_seeds), help="每个训练策略重复使用的随机种子。")
    parser.add_argument("--max_epochs", type=int, default=DIRECT_CONFIG.max_epochs, help="单个模型最大训练轮数。")
    parser.add_argument("--patience", type=int, default=DIRECT_CONFIG.patience, help="早停耐心轮数。")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=DIRECT_CONFIG.freeze_encoder_epochs, help="训练初期冻结T1/T2编码器的轮数。")
    parser.add_argument("--encoder_lr", type=float, default=DIRECT_CONFIG.encoder_lr, help="T1/T2编码器学习率。")
    parser.add_argument("--finetune_other_lr", type=float, default=DIRECT_CONFIG.finetune_other_lr, help="SIEMENS监督微调中非编码器模块学习率。")
    parser.add_argument("--ssl_only_other_lr", type=float, default=DIRECT_CONFIG.ssl_only_other_lr, help="UIH SSL-only中新初始化模块学习率。")
    parser.add_argument("--weight_decay", type=float, default=DIRECT_CONFIG.weight_decay, help="AdamW权重衰减系数。")
    parser.add_argument("--batch_size", type=int, default=DIRECT_CONFIG.batch_size, help="训练与推理批量大小。")
    parser.add_argument("--num_workers", type=int, default=DIRECT_CONFIG.num_workers, help="DataLoader工作进程数；Windows建议为0。")
    parser.add_argument("--device", default=DIRECT_CONFIG.device, help="训练设备，例如cuda或cpu。")
    parser.add_argument("--volume_size", type=int, nargs=3, default=DIRECT_CONFIG.volume_size, help="输入体尺寸，顺序为层数、高度、宽度。")
    parser.add_argument("--threshold", type=float, default=DIRECT_CONFIG.threshold, help="二分类阳性判定阈值。")
    parser.add_argument("--n_bootstrap", type=int, default=DIRECT_CONFIG.n_bootstrap, help="患者级bootstrap重复次数。")
    parser.add_argument("--reuse_existing", action=argparse.BooleanOptionalAction, default=DIRECT_CONFIG.reuse_existing, help="是否复用已有checkpoint和预测以支持断点继续。")
    parser.add_argument("--preflight_only", action=argparse.BooleanOptionalAction, default=DIRECT_CONFIG.preflight_only, help="是否只检查路径、标签和数据划分而不训练。")
    return parser


def config_from_args(args: argparse.Namespace) -> AdaptationConfig:
    return AdaptationConfig(
        uih_data_root=Path(args.uih_data_root),
        bin_source_checkpoint=Path(args.bin_source_checkpoint),
        stage_source_checkpoint=Path(args.stage_source_checkpoint),
        ssl_checkpoint=Path(args.ssl_checkpoint),
        tasks=tuple(args.tasks),
        output_root=Path(args.output_root),
        outer_folds=args.outer_folds,
        inner_val_ratio=args.inner_val_ratio,
        split_seed=args.split_seed,
        training_seeds=tuple(args.training_seeds),
        max_epochs=args.max_epochs,
        patience=args.patience,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        encoder_lr=args.encoder_lr,
        finetune_other_lr=args.finetune_other_lr,
        ssl_only_other_lr=args.ssl_only_other_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        volume_size=tuple(args.volume_size),
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        reuse_existing=args.reuse_existing,
        preflight_only=args.preflight_only,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        run_experiment(config)


if __name__ == "__main__":
    main()
