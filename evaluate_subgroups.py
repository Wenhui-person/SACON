

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
import torch
import torch.utils.data as data_utils

from data.dataset import ONFHDataset
from evaluate_external_protocol import infer_predictions, load_model_from_checkpoint


TASKS = ("bin", "stage")
SIEMENS_STRATEGY = "siemens_internal_oof"
UIH_TRANSFER_STRATEGIES = ("zero_shot", "siemens_finetune", "uih_ssl_only")
STRATEGY_LABELS = {
    SIEMENS_STRATEGY: "SIEMENS internal OOF",
    "zero_shot": "UIH zero-shot",
    "siemens_finetune": "UIH SIEMENS fine-tune",
    "uih_ssl_only": "UIH SSL-only retraining",
}
AGE_GROUP_ORDER = ("<40", "40–59", "≥60")
SEX_ORDER = ("Female", "Male")
STAGE_NAMES = ("Normal", "Early", "Late")
REQUIRED_LABEL_COLUMNS = (
    "Left Label",
    "Left Stage",
    "Right Label",
    "Right Stage",
    "Left ELS",
    "Right ELS",
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}
GROUP_COLORS = {
    "Female": COLORS["orange"],
    "Male": COLORS["blue"],
    "<40": COLORS["green"],
    "40–59": COLORS["purple"],
    "≥60": COLORS["red"],
}

BINARY_METRICS = (
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "f1",
    "brier",
)
STAGE_METRICS = (
    "qwk",
    "macro_f1",
    "balanced_accuracy",
    "mae",
    "within_one_stage_accuracy",
    "severe_cross_stage_error_rate",
)
PRIMARY_METRIC = {"bin": "auroc", "stage": "qwk"}
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1 score",
    "brier": "Brier score",
    "qwk": "Quadratic weighted kappa",
    "macro_f1": "Macro-F1",
    "mae": "Mean absolute error",
    "within_one_stage_accuracy": "Within-one-stage accuracy",
    "severe_cross_stage_error_rate": "Severe error rate",
}


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DirectRunConfig:

    run_inference: bool = True
    cohorts: Tuple[str, ...] = ("internal", "external")
    run_analyze: bool = True

    tasks: Tuple[str, ...] = ("bin", "stage")

    siemens_data_root: Path = Path(
        r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\siemens_onfh_data.json"
    )
    siemens_demographics_xlsx: Path = Path(
        r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\info_details.xlsx"
    )
    uih_data_root: Path = Path(
        r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH\uih_onfh_data_resampled.json"
    )
    uih_demographics_xlsx: Path = Path(
        r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH\info_details.xlsx"
    )
    uih_adaptation_predictions_csv: Path = (
        PROJECT_ROOT / "outputs" / "uih_adaptation" / "all_oof_predictions.csv"
    )
    uih_transfer_strategies: Tuple[str, ...] = UIH_TRANSFER_STRATEGIES
    external_results_root: Path = Path(r"E:\BackupToDriveF\myprojects\External_protocol_eval_uih_oof")

    bin_checkpoint_dir: Path = Path(
        r"G:\【重要】文章-汇总\【Article】SACON_-Symmetry-Aware Contralateral-difference Analysis-\SACON-原始结果\SACON-fastmri-ssl-二分类任务-最新结果"
    )
    bin_checkpoint_name: str = "bin_full_ssl_best.pth"
    stage_checkpoint_dir: Path = Path(
        r"G:\【重要】文章-汇总\【Article】SACON_-Symmetry-Aware Contralateral-difference Analysis-\SACON-原始结果\SACON-fastmri-ssl-分期任务-最新结果"
    )
    stage_checkpoint_name: str = "stage_full_ssl_best.pth"

    output_root: Path = PROJECT_ROOT / "outputs" / "subgroups_fastMRI_ssl"
    batch_size: int = 2
    num_workers: int = 0
    device: str = "cuda"
    threshold: float = 0.5
    volume_size: Tuple[int, int, int] = (19, 314, 224)
    complete_case: bool = False
    group_by_base_id: bool = False
    repeated_holdout_7_3: bool = False
    real_only_test: bool = True
    n_bootstrap: int = 2000
    seed: int = 2026
    reuse_existing_predictions: bool = True


DIRECT_RUN_CONFIG = DirectRunConfig()


@dataclass
class LogisticFit:

    beta: np.ndarray
    covariance: np.ndarray
    columns: List[str]
    converged: bool
    penalized: bool = False


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.weight": "normal",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": COLORS["black"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
            "axes.labelpad": 5,
            "legend.fontsize": 8,
            "legend.title_fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.formatter.useoffset": False,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: Optional[str] = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E8EBED", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def fit_metric_axis(ax: plt.Axes, values: np.ndarray, metric: str) -> None:
    from matplotlib.ticker import MaxNLocator
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return
    domain = (-1.0, 1.0) if metric == "qwk" else (0.0, 2.0 if metric == "mae" else 1.0)
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 0.12)
    center = (low + high) / 2
    margin = span * 0.12
    ax.set_xlim(max(domain[0] - .02, center - span / 2 - margin),
                min(domain[1] + .02, center + span / 2 + margin))
    locator = MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    ax.xaxis.set_major_locator(locator)
    left, right = ax.get_xlim()
    ax.set_xticks([tick for tick in locator.tick_values(left, right)
                  if left <= tick <= right and domain[0] <= tick <= domain[1]])


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        is_panel = artist.get_text() in tuple("abcdefghijklmnopqrstuvwxyz") and artist.get_fontweight() == "bold"
        artist.set_fontsize(8 if is_panel else min(9, max(5, artist.get_fontsize())))
        if not is_panel:
            artist.set_fontweight("normal")
    for ax in fig.axes:
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for label in (*ax.get_xticklabels(which="both"), *ax.get_yticklabels(which="both")):
            label.set_fontweight("bold")
        ax.xaxis.get_offset_text().set_fontweight("bold")
        ax.yaxis.get_offset_text().set_fontweight("bold")
        for location in ("left", "center", "right"):
            title = ax.get_title(loc=location)
            if title:
                ax.set_title(title, loc=location, fontweight="bold")
    for artist in fig.findobj(Text):
        if artist.get_gid() == "subgroup-title":
            artist.set_fontweight("bold")
    if fig._suptitle is not None:
        fig._suptitle.set_fontweight("bold")
    from matplotlib.collections import LineCollection
    for line in fig.findobj(Line2D):
        line.set_linewidth(line.get_linewidth() * 1.25)
    for collection in fig.findobj(LineCollection):
        collection.set_linewidth(collection.get_linewidths() * 1.25)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(spine.get_linewidth() * 1.25)
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    for legend in fig.legends:
        if legend.get_gid() == "age-effect-legend":
            renderer = fig.canvas.get_renderer()
            bottom = min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in fig.axes)
            anchor_y = fig.transFigure.inverted().transform((0, bottom - fig.dpi * 8 / 72))[1]
            legend.set_bbox_to_anchor((0.5, anchor_y), transform=fig.transFigure)
            fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    column_width = (89 if fig.get_figwidth() <= 3.6 else 183) / 25.4
    width = max(column_width, bounds.width + 0.16)
    export_bounds = Bbox.from_bounds(
        bounds.x0 - (width - bounds.width) / 2,
        bounds.y0 - 0.08, width, bounds.height + 0.16,
    )
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches=export_bounds)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches=export_bounds)
    plt.close(fig)


def normalize_id(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def find_id_column(df: pd.DataFrame) -> str:
    candidates = ("case_id", "ID", "id", "Case ID", "patient_id", "Patient ID")
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"人口学表中找不到患者 ID 列。现有列：{list(df.columns)}")


def find_column(df: pd.DataFrame, candidates: Sequence[str], display_name: str) -> str:
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return str(lookup[candidate.lower()])
    raise KeyError(f"人口学表中找不到 {display_name} 列。现有列：{list(df.columns)}")


def normalize_sex(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    female = {"女", "f", "female", "woman", "0"}
    male = {"男", "m", "male", "man", "1"}
    if text in female:
        return "Female"
    if text in male:
        return "Male"
    return None


def normalize_age(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).strip())
    if not match:
        return np.nan
    age = float(match.group())
    return age if 0.0 <= age <= 120.0 else np.nan


def assign_age_group(age: object) -> Optional[str]:
    if pd.isna(age):
        return None
    age = float(age)
    if age < 40:
        return "<40"
    if age < 60:
        return "40–59"
    return "≥60"


def _n_unique_non_missing(series: pd.Series) -> int:
    return int(series.dropna().astype(str).str.strip().nunique())


def load_demographics(
    xlsx_path: str,
    relevant_ids: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(xlsx_path, header=0)
    id_column = find_id_column(raw)
    sex_column = find_column(raw, ("Sex", "Gender", "性别"), "性别")
    age_column = find_column(raw, ("Age", "年龄"), "年龄")
    raw["patient_id"] = raw[id_column].map(normalize_id)

    if relevant_ids is not None:
        relevant_set = {str(value) for value in relevant_ids}
        raw = raw[raw["patient_id"].isin(relevant_set)].copy()

    conflicts = raw.groupby("patient_id", dropna=True).agg(
        source_rows=("patient_id", "size"),
        sex_distinct=(sex_column, _n_unique_non_missing),
        age_distinct=(age_column, _n_unique_non_missing),
    )
    conflicts = conflicts.reset_index()
    conflicts["sex_conflict"] = conflicts["sex_distinct"] > 1
    conflicts["age_conflict"] = conflicts["age_distinct"] > 1

    available_labels = [column for column in REQUIRED_LABEL_COLUMNS if column in raw.columns]
    selected = raw.copy()
    if len(available_labels) == len(REQUIRED_LABEL_COLUMNS):
        selected = selected.dropna(subset=["patient_id", *available_labels]).copy()
        for column in available_labels:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
        selected = selected.dropna(subset=available_labels)
    selected = selected.drop_duplicates("patient_id", keep="first")

    demographics = pd.DataFrame(
        {
            "patient_id": selected["patient_id"].astype(str),
            "sex": selected[sex_column].map(normalize_sex),
            "age": selected[age_column].map(normalize_age),
        }
    )
    demographics["age_group"] = demographics["age"].map(assign_age_group)
    demographics["age_group"] = pd.Categorical(
        demographics["age_group"], categories=AGE_GROUP_ORDER, ordered=True
    )
    return demographics, conflicts


def expand_checkpoints(values: Optional[Sequence[str]]) -> List[str]:
    paths: List[str] = []
    for value in values or []:
        matches = sorted(glob.glob(value)) if any(char in value for char in "*?[]") else [value]
        paths.extend(matches)
    return [str(Path(path)) for path in paths]


def extract_fold_number(path: str) -> Optional[int]:
    matches = re.findall(r"(?:^|[\\/_.-])fold[_-]?(\d+)(?=$|[\\/_.-])", str(path), flags=re.IGNORECASE)
    if not matches:
        return None
    unique_folds = {int(value) for value in matches}
    if len(unique_folds) > 1:
        raise ValueError(f"checkpoint 路径包含多个不同 fold 编号：{path}")
    return unique_folds.pop()


def validate_checkpoints(checkpoints: Sequence[str], task: str) -> List[Tuple[int, str]]:
    if not checkpoints:
        raise ValueError(f"{task} 至少需要 1 个 checkpoint。")
    missing = [path for path in checkpoints if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{task} checkpoint 不存在：{missing}")

    fold_pairs: List[Tuple[int, str]] = []
    for position, path in enumerate(checkpoints):
        fold = extract_fold_number(path)
        if fold is None and len(checkpoints) == 5:
            fold = position
        if fold is None:
            raise ValueError(
                f"无法从 checkpoint 路径识别 fold 编号：{path}。"
                "部分折运行时，目录名或文件名必须包含 fold_0 至 fold_4。"
            )
        if fold not in range(5):
            raise ValueError(f"{task} fold 编号必须位于 0--4，当前为 fold {fold}：{path}")
        fold_pairs.append((fold, str(Path(path))))

    folds = [fold for fold, _ in fold_pairs]
    if len(folds) != len(set(folds)):
        raise ValueError(f"{task} 存在重复 fold checkpoint：{folds}")
    return sorted(fold_pairs, key=lambda item: item[0])


def discover_fold_checkpoints(directory: Path, filename: str, task: str) -> List[str]:
    if not directory.is_dir():
        return []
    candidates = [str(path) for path in directory.rglob(filename) if path.is_file()]
    if not candidates:
        return []
    return [path for _, path in validate_checkpoints(candidates, task)]


def all_dataset_indices(data_root: str) -> List[int]:
    with open(data_root, "r", encoding="utf-8") as handle:
        samples = json.load(handle, object_pairs_hook=OrderedDict)
    return list(range(len(samples["patient_id"])))


def get_base_patient_id(patient_id: object) -> str:
    return re.sub(r"_aug_\d+$", "", str(patient_id))


def get_fold_stratify_labels(data: Mapping[str, Sequence[object]], task: str) -> List[str]:
    if task == "bin":
        return [f"{left}_{right}" for left, right in zip(data["label_left_bin"], data["label_right_bin"])]
    return [f"{left}_{right}" for left, right in zip(data["label_left_ELS"], data["label_right_ELS"])]


def _validate_base_id_labels(data: Mapping[str, Sequence[object]], base_ids: Sequence[str]) -> None:
    labels_by_id: Dict[str, Tuple[object, ...]] = {}
    for index, base_id in enumerate(base_ids):
        labels = (
            data["label_left_bin"][index],
            data["label_right_bin"][index],
            data["label_left_ELS"][index],
            data["label_right_ELS"][index],
        )
        if base_id in labels_by_id and labels_by_id[base_id] != labels:
            raise ValueError(f"基础患者 ID {base_id} 存在不一致标签：{labels_by_id[base_id]} vs {labels}")
        labels_by_id[base_id] = labels


def is_synthetic_sample(patient_id: object) -> bool:
    return str(patient_id).startswith("gen_full_")


def get_stable_real_stratify_labels(
    data: Mapping[str, Sequence[object]],
    task: str,
    real_indices: Sequence[int],
    n_splits: int,
) -> List[str]:
    combined = get_fold_stratify_labels(data, task)
    selected = [combined[index] for index in real_indices]
    counts = pd.Series(selected).value_counts().to_dict()
    if min(counts.values()) >= n_splits:
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
    if min(fallback_counts.values()) < n_splits:
        raise ValueError(f"降级分层后最小类别仍少于{n_splits}例：{fallback_counts}")
    print(f"[CV] 双侧标签组合存在稀有类别，改用{fallback_name}分层：{fallback_counts}")
    return selected


def build_real_only_test_splits(
    data: Mapping[str, Sequence[object]],
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    patient_ids = list(data["patient_id"])
    labels = get_fold_stratify_labels(data, task)
    synthetic_indices = np.asarray(
        [index for index, patient_id in enumerate(patient_ids) if is_synthetic_sample(patient_id)],
        dtype=np.int64,
    )
    real_indices = np.asarray(
        [index for index, patient_id in enumerate(patient_ids) if not is_synthetic_sample(patient_id)],
        dtype=np.int64,
    )
    split_mode = (
        "real-only-test-with-synthetic-training"
        if len(synthetic_indices)
        else "standard-stratified-five-fold"
    )
    print(
        f"[CV] split_mode={split_mode} | real={len(real_indices)} "
        f"synthetic={len(synthetic_indices)}"
    )
    if len(real_indices) < n_splits:
        raise ValueError(f"真实样本数{len(real_indices)}少于折数{n_splits}。")
    real_labels = get_stable_real_stratify_labels(data, task, real_indices, n_splits)
    label_counts = pd.Series(real_labels).value_counts().sort_index().to_dict()
    if min(label_counts.values()) < n_splits:
        raise ValueError(f"真实数据最小分层类别少于{n_splits}例：{label_counts}")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold, (real_train_position, real_test_position) in enumerate(
        splitter.split(real_indices, real_labels)
    ):
        train_index = np.concatenate(
            [real_indices[real_train_position], synthetic_indices]
        ).astype(np.int64)
        test_index = real_indices[real_test_position].astype(np.int64)
        if any(is_synthetic_sample(patient_ids[index]) for index in test_index):
            raise RuntimeError(f"fold {fold}测试索引中包含合成样本。")
        splits.append((train_index, test_index))
    return splits


def test_ids_from_saved_split(
    checkpoint: str,
    reconstructed_ids: Sequence[int],
    data: Mapping[str, Sequence[object]],
) -> Tuple[np.ndarray, str]:
    split_path = Path(checkpoint).parent / "fold_data_split.csv"
    reconstructed = np.asarray(reconstructed_ids, dtype=np.int64)
    if not split_path.is_file():
        return reconstructed, "reconstructed"

    table = pd.read_csv(split_path)
    required = {"split", "source_index", "patient_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"划分CSV缺少字段{sorted(missing)}：{split_path}")
    test_table = table[table["split"].astype(str).str.lower() == "test"].copy()
    if test_table.empty:
        raise ValueError(f"划分CSV中没有test记录：{split_path}")
    saved = test_table["source_index"].astype(int).to_numpy(np.int64)
    if saved.min() < 0 or saved.max() >= len(data["patient_id"]):
        raise IndexError(f"划分CSV包含越界source_index：{split_path}")
    expected_ids = [str(data["patient_id"][index]) for index in saved]
    if expected_ids != test_table["patient_id"].astype(str).tolist():
        raise ValueError(f"划分CSV与当前数据清单的patient_id顺序不一致：{split_path}")
    if any(is_synthetic_sample(patient_id) for patient_id in expected_ids):
        raise ValueError(f"划分CSV的测试集中检测到合成样本：{split_path}")
    if set(saved.tolist()) != set(reconstructed.tolist()):
        raise ValueError(f"划分CSV与evaluate_subgroups.py重建的测试索引不一致：{split_path}")
    return saved, str(split_path)


def build_cv_splits(
    data: Mapping[str, Sequence[object]],
    task: str,
    group_by_base_id: bool,
    repeated_holdout_7_3: bool = False,
    real_only_test: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    patient_ids = list(data["patient_id"])
    if real_only_test:
        if group_by_base_id or repeated_holdout_7_3:
            raise ValueError(
                "real_only_test不能与group_by_base_id或repeated_holdout_7_3同时启用。"
            )
        return build_real_only_test_splits(data, task)
    if repeated_holdout_7_3:
        base_ids = [get_base_patient_id(patient_id) for patient_id in patient_ids]
        labels = get_fold_stratify_labels(data, task)
        _validate_base_id_labels(data, base_ids)
        group_indices: Dict[str, List[int]] = {}
        group_labels: Dict[str, str] = {}
        for index, (base_id, label) in enumerate(zip(base_ids, labels)):
            group_indices.setdefault(base_id, []).append(index)
            group_labels.setdefault(base_id, label)
        unique_ids = list(group_indices)
        unique_labels = [group_labels[base_id] for base_id in unique_ids]
        splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_groups, validation_groups in splitter.split(unique_ids, unique_labels):
            train_ids = {unique_ids[index] for index in train_groups}
            validation_ids = {unique_ids[index] for index in validation_groups}
            train_index = np.asarray(
                [index for base_id in unique_ids if base_id in train_ids for index in group_indices[base_id]],
                dtype=np.int64,
            )
            validation_index = np.asarray(
                [index for base_id in unique_ids if base_id in validation_ids for index in group_indices[base_id]],
                dtype=np.int64,
            )
            splits.append((train_index, validation_index))
        return splits

    if not group_by_base_id:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        return [(train.astype(np.int64), validation.astype(np.int64)) for train, validation in splitter.split(patient_ids)]

    base_ids = [get_base_patient_id(patient_id) for patient_id in patient_ids]
    labels = get_fold_stratify_labels(data, task)
    _validate_base_id_labels(data, base_ids)
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(patient_ids, labels, groups=base_ids))
    except ValueError:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(patient_ids, labels, groups=base_ids))
    return [(train.astype(np.int64), validation.astype(np.int64)) for train, validation in splits]


def make_loader(
    data_root: str,
    ids: Sequence[int],
    volume_size: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
) -> Tuple[ONFHDataset, data_utils.DataLoader]:
    dataset = ONFHDataset(
        data_root=data_root,
        ids=list(ids),
        volume_size=volume_size,
        augment=False,
        online_train_augment=False,
    )
    loader = data_utils.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, loader


def prediction_columns_ok(df: pd.DataFrame, task: str) -> None:
    required = {"patient_id", "side", "label", "prediction", "task"}
    if task == "bin":
        required.add("prob_positive")
    else:
        required.update({"prob_class_0", "prob_class_1", "prob_class_2"})
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{task} 预测表缺少字段：{sorted(missing)}")


def run_internal_oof_for_task(
    task: str,
    checkpoints: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    from utils.subgroup_reporting import infer_saved_oof
    return infer_saved_oof(sys.modules[__name__], task, checkpoints, args, device)



def _ensemble_task_predictions(fold_predictions: pd.DataFrame, task: str, threshold: float) -> pd.DataFrame:
    keys = ["cohort", "task", "patient_id", "source_index", "side", "label"]
    grouped = fold_predictions.groupby(keys, as_index=False, dropna=False)
    if task == "bin":
        result = grouped["prob_positive"].mean()
        result["prediction"] = (result["prob_positive"] >= threshold).astype(int)
        result["confidence"] = np.maximum(result["prob_positive"], 1.0 - result["prob_positive"])
    else:
        probability_columns = [f"prob_class_{idx}" for idx in range(3)]
        result = grouped[probability_columns].mean()
        probabilities = result[probability_columns].to_numpy(float)
        probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
        result[probability_columns] = probabilities
        result["prediction"] = probabilities.argmax(axis=1)
        result["confidence"] = probabilities.max(axis=1)
        result["abs_error"] = np.abs(result["prediction"] - result["label"]).astype(int)
    result["is_error"] = result["prediction"].astype(int) != result["label"].astype(int)
    result["ensemble_size"] = fold_predictions["fold"].nunique()
    return result


def run_external_ensemble_for_task(
    task: str,
    checkpoints: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    fold_checkpoints = validate_checkpoints(checkpoints, task)
    ids = all_dataset_indices(args.data_root)
    dataset, loader = make_loader(
        args.data_root,
        ids,
        tuple(args.volume_size),
        args.batch_size,
        args.num_workers,
    )
    predictions: List[pd.DataFrame] = []
    for fold, checkpoint in fold_checkpoints:
        model, checkpoint_data = load_model_from_checkpoint(checkpoint, task, device)
        num_classes = int(checkpoint_data.get("train_cfg", {}).get("num_stages", 3))
        fold_df = infer_predictions(
            model=model,
            loader=loader,
            dataset=dataset,
            task=task,
            checkpoint_name=Path(checkpoint).stem,
            device=device,
            num_classes=num_classes,
            threshold=args.threshold,
        )
        fold_df["fold"] = fold
        fold_df["cohort"] = args.cohort
        predictions.append(fold_df)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _ensemble_task_predictions(pd.concat(predictions, ignore_index=True), task, args.threshold)


def attach_demographics(
    predictions: pd.DataFrame,
    demographics_xlsx: str,
    complete_case: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predictions.copy()
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["demographic_id"] = predictions["patient_id"].map(get_base_patient_id)
    demographics, conflicts = load_demographics(
        demographics_xlsx,
        relevant_ids=predictions["demographic_id"].unique(),
    )
    demographics = demographics.rename(columns={"patient_id": "demographic_id"})
    merged = predictions.merge(demographics, on="demographic_id", how="left", validate="many_to_one")

    audit_rows: List[Dict[str, object]] = []
    for cohort, group in merged.groupby("cohort", dropna=False):
        patient_view = group.drop_duplicates("patient_id")
        conflict_ids = set(conflicts.loc[conflicts["sex_conflict"] | conflicts["age_conflict"], "patient_id"])
        audit_rows.append(
            {
                "cohort": cohort,
                "n_prediction_rows": len(group),
                "n_patients": patient_view["patient_id"].nunique(),
                "n_missing_sex": int(patient_view["sex"].isna().sum()),
                "n_missing_age": int(patient_view["age"].isna().sum()),
                "n_source_conflict_ids": int(patient_view["demographic_id"].isin(conflict_ids).sum()),
                "complete_case_patients": int(patient_view[["sex", "age"]].notna().all(axis=1).sum()),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if complete_case:
        merged = merged.dropna(subset=["sex", "age", "age_group"]).copy()
    return merged, audit


def standardize_strategy_column(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    if "strategy" not in result.columns:
        cohort_text = result["cohort"].astype(str).str.lower()
        result["strategy"] = np.where(
            cohort_text.str.contains("uih"),
            "zero_shot",
            SIEMENS_STRATEGY,
        )
    result["strategy"] = result["strategy"].astype(str)
    return result


def validate_uih_strategy_pairing(predictions: pd.DataFrame) -> None:
    key_columns = ["task", "patient_id", "side"]
    reference: Optional[set] = None
    reference_strategy: Optional[str] = None
    for strategy, group in predictions.groupby("strategy", sort=False):
        if group.duplicated(key_columns).any():
            raise ValueError(f"UIH策略 {strategy} 存在重复的任务-患者-侧别预测。")
        keys = set(map(tuple, group[key_columns].astype(str).to_numpy()))
        if reference is None:
            reference = keys
            reference_strategy = str(strategy)
        elif keys != reference:
            missing = len(reference - keys)
            extra = len(keys - reference)
            raise ValueError(
                f"UIH策略 {strategy} 与 {reference_strategy} 的预测覆盖不一致："
                f"缺少{missing}条，多出{extra}条。"
            )


def run_uih_adaptation_command(args: argparse.Namespace) -> None:
    if getattr(args, "external_results_root", None):
        from utils.subgroup_reporting import import_external
        import_external(sys.modules[__name__], args)
        return
    prediction_path = Path(args.adaptation_prediction_csv)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"UIH迁移预测CSV不存在：{prediction_path}")

    predictions = pd.read_csv(prediction_path)
    required = {"task", "strategy", "patient_id", "side", "label", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"UIH迁移预测CSV缺少必要字段：{sorted(missing)}")

    selected = tuple(dict.fromkeys(args.strategies))
    unknown = set(selected) - set(UIH_TRANSFER_STRATEGIES)
    if unknown:
        raise ValueError(
            f"未知UIH迁移策略：{sorted(unknown)}；允许值为{list(UIH_TRANSFER_STRATEGIES)}。"
        )
    predictions = predictions[predictions["strategy"].astype(str).isin(selected)].copy()
    selected_tasks = tuple(dict.fromkeys(args.tasks))
    predictions = predictions[predictions["task"].astype(str).isin(selected_tasks)].copy()
    if predictions.empty:
        raise ValueError("所选UIH迁移策略在预测CSV中均不存在。")
    absent = set(selected) - set(predictions["strategy"].astype(str))
    if absent:
        raise ValueError(f"UIH迁移预测CSV缺少所选策略：{sorted(absent)}")
    absent_tasks = set(selected_tasks) - set(predictions["task"].astype(str))
    if absent_tasks:
        raise ValueError(f"UIH迁移预测CSV缺少所选任务：{sorted(absent_tasks)}")

    predictions["cohort"] = "UIH_external_cross_protocol"
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["prediction"] = predictions["prediction"].astype(int)
    predictions["label"] = predictions["label"].astype(int)
    bin_mask = predictions["task"].astype(str) == "bin"
    if bin_mask.any():
        if "prob_positive" not in predictions.columns:
            raise ValueError("UIH二分类迁移预测缺少prob_positive字段。")
        predictions.loc[bin_mask, "prediction"] = (
            predictions.loc[bin_mask, "prob_positive"].astype(float) >= args.threshold
        ).astype(int)
    predictions["is_error"] = predictions["prediction"] != predictions["label"]
    validate_uih_strategy_pairing(predictions)

    predictions, audit = attach_demographics(
        predictions,
        args.demographics_xlsx,
        args.complete_case,
    )
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "subgroup_predictions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "demographic_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "mode": "uih-supervised-adaptation",
        "cohort": "UIH_external_cross_protocol",
        "source_prediction_csv": str(prediction_path),
        "demographics_xlsx": args.demographics_xlsx,
        "strategies": list(selected),
        "tasks": list(selected_tasks),
        "threshold": args.threshold,
        "complete_case": args.complete_case,
        "real_only_test": False,
        "n_rows": len(predictions),
        "n_patients": predictions["patient_id"].nunique(),
        "tasks_present": sorted(predictions["task"].astype(str).unique().tolist()),
        "interpretation_boundary": (
            "zero_shot属于外部跨设备协议评估；siemens_finetune和uih_ssl_only属于UIH目标域有监督适配"
        ),
    }
    with (output_dir / "subgroup_inference_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[OK] UIH迁移策略亚组预测已保存：{output_dir / 'subgroup_predictions.csv'}")


def binary_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["label"].to_numpy(int)
    y_prob = frame["prob_positive"].to_numpy(float)
    y_pred = frame["prediction"].to_numpy(int)
    result = {metric: np.nan for metric in BINARY_METRICS}
    if len(frame) == 0:
        return result
    if np.unique(y_true).size == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
        result["auprc"] = float(average_precision_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result["sensitivity"] = float(tp / (tp + fn)) if tp + fn else np.nan
    result["specificity"] = float(tn / (tn + fp)) if tn + fp else np.nan
    result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size == 2 else np.nan
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    result["brier"] = float(brier_score_loss(y_true, y_prob))
    return result


def stage_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["label"].to_numpy(int)
    y_pred = frame["prediction"].to_numpy(int)
    result = {metric: np.nan for metric in STAGE_METRICS}
    if len(frame) == 0:
        return result
    error = np.abs(y_pred - y_true)
    result.update(
        {
            "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2])),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mae": float(error.mean()),
            "within_one_stage_accuracy": float((error <= 1).mean()),
            "severe_cross_stage_error_rate": float((error > 1).mean()),
        }
    )
    return result


def compute_metrics(frame: pd.DataFrame, task: str) -> Dict[str, float]:
    return binary_metrics(frame) if task == "bin" else stage_metrics(frame)


def cluster_bootstrap_metric_samples(
    frame: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    patient_groups = {pid: group for pid, group in frame.groupby("patient_id", sort=False)}
    patients = np.asarray(list(patient_groups), dtype=object)
    metric_names = BINARY_METRICS if task == "bin" else STAGE_METRICS
    samples = {metric: np.full(n_bootstrap, np.nan, dtype=float) for metric in metric_names}
    if len(patients) == 0:
        return samples
    for iteration in range(n_bootstrap):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([patient_groups[pid] for pid in chosen], ignore_index=True)
        values = compute_metrics(boot, task)
        for metric, value in values.items():
            samples[metric][iteration] = value
    return samples


def percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 20:
        return np.nan, np.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.where(np.isfinite(values))[0]
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def summarize_subgroups(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    gap_rows: List[Dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(seed)
    rngs = iter(np.random.default_rng(child) for child in seed_sequence.spawn(10000))

    for (cohort, strategy, task), task_frame in predictions.groupby(
        ["cohort", "strategy", "task"], sort=False
    ):
        metric_names = BINARY_METRICS if task == "bin" else STAGE_METRICS
        for variable, levels in (("sex", SEX_ORDER), ("age_group", AGE_GROUP_ORDER)):
            bootstrap_by_level: Dict[str, Dict[str, np.ndarray]] = {}
            for level in levels:
                group = task_frame[task_frame[variable].astype(str) == level]
                if group.empty:
                    continue
                estimates = compute_metrics(group, task)
                bootstrap = cluster_bootstrap_metric_samples(group, task, n_bootstrap, next(rngs))
                bootstrap_by_level[level] = bootstrap
                for metric in metric_names:
                    ci_low, ci_high = percentile_ci(bootstrap[metric])
                    metric_rows.append(
                        {
                            "cohort": cohort,
                            "strategy": strategy,
                            "task": task,
                            "subgroup_variable": variable,
                            "subgroup": level,
                            "metric": metric,
                            "estimate": estimates[metric],
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "n_patients": group["patient_id"].nunique(),
                            "n_hips": len(group),
                            "n_positive_or_non_normal": int((group["label"] > 0).sum()),
                        }
                    )

            comparisons = [("Male", "Female")] if variable == "sex" else [("<40", "40–59"), ("≥60", "40–59")]
            for level, reference in comparisons:
                if level not in bootstrap_by_level or reference not in bootstrap_by_level:
                    continue
                point_level = compute_metrics(task_frame[task_frame[variable].astype(str) == level], task)
                point_reference = compute_metrics(task_frame[task_frame[variable].astype(str) == reference], task)
                for metric in metric_names:
                    difference = bootstrap_by_level[level][metric] - bootstrap_by_level[reference][metric]
                    finite = difference[np.isfinite(difference)]
                    ci_low, ci_high = percentile_ci(finite)
                    p_value = np.nan
                    if finite.size:
                        p_value = min(1.0, 2.0 * min(np.mean(finite <= 0), np.mean(finite >= 0)))
                    gap_rows.append(
                        {
                            "cohort": cohort,
                            "strategy": strategy,
                            "task": task,
                            "subgroup_variable": variable,
                            "comparison": f"{level} vs {reference}",
                            "metric": metric,
                            "difference": point_level[metric] - point_reference[metric],
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_value,
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    gaps_df = pd.DataFrame(gap_rows)
    if not gaps_df.empty:
        gaps_df["p_holm"] = np.nan
        for _, index in gaps_df.groupby(["cohort", "strategy", "task"], sort=False).groups.items():
            gaps_df.loc[index, "p_holm"] = holm_adjust(gaps_df.loc[index, "p_value"].to_numpy(float))
    return metrics_df, gaps_df


def summarize_transfer_strategy_differences(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    uih = predictions[predictions["cohort"].astype(str).str.lower().str.contains("uih")].copy()
    available = set(uih["strategy"].astype(str))
    comparisons = [
        ("siemens_finetune", "zero_shot"),
        ("uih_ssl_only", "zero_shot"),
        ("siemens_finetune", "uih_ssl_only"),
    ]
    comparisons = [(left, right) for left, right in comparisons if {left, right} <= available]
    if not comparisons:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(seed + 7919)
    rngs = iter(np.random.default_rng(child) for child in seed_sequence.spawn(1000))
    for task, task_frame in uih.groupby("task", sort=False):
        metric_names = BINARY_METRICS if task == "bin" else STAGE_METRICS
        for variable, levels in (("sex", SEX_ORDER), ("age_group", AGE_GROUP_ORDER)):
            for level in levels:
                subgroup = task_frame[task_frame[variable].astype(str) == level]
                if subgroup.empty:
                    continue
                for left, right in comparisons:
                    left_frame = subgroup[subgroup["strategy"] == left]
                    right_frame = subgroup[subgroup["strategy"] == right]
                    common_patients = sorted(
                        set(left_frame["patient_id"].astype(str))
                        & set(right_frame["patient_id"].astype(str))
                    )
                    if not common_patients:
                        continue
                    left_frame = left_frame[left_frame["patient_id"].astype(str).isin(common_patients)]
                    right_frame = right_frame[right_frame["patient_id"].astype(str).isin(common_patients)]
                    left_groups = {
                        str(patient): group
                        for patient, group in left_frame.groupby(left_frame["patient_id"].astype(str), sort=False)
                    }
                    right_groups = {
                        str(patient): group
                        for patient, group in right_frame.groupby(right_frame["patient_id"].astype(str), sort=False)
                    }
                    left_point = compute_metrics(left_frame, str(task))
                    right_point = compute_metrics(right_frame, str(task))
                    samples = {
                        metric: np.full(n_bootstrap, np.nan, dtype=float)
                        for metric in metric_names
                    }
                    rng = next(rngs)
                    for iteration in range(n_bootstrap):
                        chosen = rng.choice(common_patients, size=len(common_patients), replace=True)
                        left_boot = pd.concat([left_groups[str(patient)] for patient in chosen], ignore_index=True)
                        right_boot = pd.concat([right_groups[str(patient)] for patient in chosen], ignore_index=True)
                        left_values = compute_metrics(left_boot, str(task))
                        right_values = compute_metrics(right_boot, str(task))
                        for metric in metric_names:
                            samples[metric][iteration] = left_values[metric] - right_values[metric]

                    for metric in metric_names:
                        finite = samples[metric][np.isfinite(samples[metric])]
                        ci_low, ci_high = percentile_ci(finite)
                        p_value = np.nan
                        if finite.size:
                            p_value = min(
                                1.0,
                                2.0 * min(float(np.mean(finite <= 0)), float(np.mean(finite >= 0))),
                            )
                        rows.append(
                            {
                                "cohort": "UIH_external_cross_protocol",
                                "task": task,
                                "subgroup_variable": variable,
                                "subgroup": level,
                                "comparison": f"{left} vs {right}",
                                "metric": metric,
                                "difference": left_point[metric] - right_point[metric],
                                "ci_low": ci_low,
                                "ci_high": ci_high,
                                "p_value": p_value,
                                "n_paired_patients": len(common_patients),
                            }
                        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = np.nan
        for _, index in result.groupby(["task", "metric"], sort=False).groups.items():
            result.loc[index, "p_holm"] = holm_adjust(result.loc[index, "p_value"].to_numpy(float))
    return result


def logistic_cluster_fit(frame: pd.DataFrame) -> Optional[LogisticFit]:
    clean = frame.dropna(subset=["age", "sex", "side", "label", "patient_id"]).copy()
    if clean.empty or clean["is_error"].nunique() < 2:
        return None
    age_center = float(clean["age"].mean())
    matrix = {
        "Intercept": np.ones(len(clean)),
        "Age (per 10 years)": (clean["age"].to_numpy(float) - age_center) / 10.0,
        "Male": (clean["sex"].astype(str) == "Male").astype(float).to_numpy(),
        "Right hip": (clean["side"].astype(str) == "right").astype(float).to_numpy(),
    }
    for label in sorted(clean["label"].astype(int).unique())[1:]:
        matrix[f"True label {label}"] = (
            clean["label"].astype(int) == label
        ).astype(float).to_numpy()
    columns = list(matrix)
    x = np.column_stack([matrix[column] for column in columns]).astype(float)
    y = clean["is_error"].astype(int).to_numpy(float)
    if np.linalg.matrix_rank(x) < x.shape[1] or clean.patient_id.nunique() <= x.shape[1]:
        return None

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = x @ beta
        probability = expit(eta)
        loss = np.logaddexp(0.0, eta).sum() - np.dot(y, eta)
        gradient = x.T @ (probability - y)
        return float(loss), gradient

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=np.zeros(x.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
    )
    beta = result.x
    penalized = False
    penalty = 0.0
    if not result.success or not np.isfinite(beta).all() or np.max(np.abs(beta)) > 20:
        penalty = 0.5
        penalty_mask = np.ones(x.shape[1], dtype=float)
        penalty_mask[0] = 0.0

        def penalized_objective(candidate: np.ndarray) -> Tuple[float, np.ndarray]:
            loss, gradient = objective(candidate)
            loss += 0.5 * penalty * float(
                np.dot(penalty_mask * candidate, candidate)
            )
            gradient = gradient + penalty * penalty_mask * candidate
            return loss, gradient

        result = minimize(
            fun=lambda candidate: penalized_objective(candidate)[0],
            x0=np.zeros(x.shape[1]),
            jac=lambda candidate: penalized_objective(candidate)[1],
            method="BFGS",
        )
        beta = result.x
        penalized = True
        if not result.success or not np.isfinite(beta).all():
            return None

    probability = expit(x @ beta)
    weight = np.clip(probability * (1.0 - probability), 1e-8, None)
    penalty_hessian = np.diag(
        np.r_[0.0, np.repeat(penalty, x.shape[1] - 1)]
    )
    bread = np.linalg.pinv(
        x.T @ (x * weight[:, None]) + penalty_hessian
    )
    score = x * (y - probability)[:, None]
    clusters = clean["patient_id"].astype(str).to_numpy()
    meat = np.zeros((x.shape[1], x.shape[1]))
    for patient in np.unique(clusters):
        cluster_score = score[clusters == patient].sum(axis=0)[:, None]
        meat += cluster_score @ cluster_score.T
    covariance = bread @ meat @ bread
    n_clusters = np.unique(clusters).size
    if n_clusters > 1:
        covariance *= n_clusters / (n_clusters - 1)
    fit = LogisticFit(
        beta=beta,
        covariance=covariance,
        columns=columns,
        converged=bool(result.success),
        penalized=penalized,
    )
    setattr(fit, "age_center", age_center)
    setattr(fit, "design_frame", clean)
    return fit

def regression_summary(
    predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], LogisticFit]]:
    rows: List[Dict[str, object]] = []
    fits: Dict[Tuple[str, str, str], LogisticFit] = {}
    for (cohort, strategy, task), group in predictions.groupby(
        ["cohort", "strategy", "task"], sort=False
    ):
        fit = logistic_cluster_fit(group)
        if fit is None:
            continue
        fits[(str(cohort), str(strategy), str(task))] = fit
        if fit.penalized:
            continue
        for term in ("Age (per 10 years)", "Male"):
            if term not in fit.columns:
                continue
            index = fit.columns.index(term)
            coefficient = fit.beta[index]
            standard_error = math.sqrt(max(float(fit.covariance[index, index]), 0.0))
            z_value = coefficient / standard_error if standard_error > 0 else np.nan
            p_value = 2.0 * norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
            rows.append(
                {
                    "cohort": cohort,
                    "strategy": strategy,
                    "task": task,
                    "term": term,
                    "odds_ratio": float(np.exp(coefficient)),
                    "ci_low": float(np.exp(coefficient - 1.96 * standard_error)),
                    "ci_high": float(np.exp(coefficient + 1.96 * standard_error)),
                    "p_value": p_value,
                    "converged": fit.converged,
                    "n_patients": fit.design_frame["patient_id"].nunique(),
                    "n_hips": len(fit.design_frame),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["p_holm"] = np.nan
        for _, index in summary.groupby(["cohort", "strategy", "task"], sort=False).groups.items():
            summary.loc[index, "p_holm"] = holm_adjust(summary.loc[index, "p_value"].to_numpy(float))
    return summary, fits


def analysis_group_label(cohort: object, strategy: object) -> str:
    strategy_text = str(strategy)
    return STRATEGY_LABELS.get(strategy_text, f"{cohort} | {strategy_text}")


def plot_demographic_composition(predictions: pd.DataFrame, figure_dir: Path) -> None:
    patient = predictions.drop_duplicates(["cohort", "patient_id"])
    cohorts = list(patient["cohort"].astype(str).drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    bins = np.arange(0, max(90, int(patient["age"].max()) + 10), 10)
    for index, cohort in enumerate(cohorts):
        values = patient.loc[patient["cohort"].astype(str) == cohort, "age"].dropna()
        axes[0].hist(values, bins=bins, histtype="step", linewidth=1.4, color=list(COLORS.values())[index % 6], label=f"{cohort} (n={len(values)})")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Patients")
    axes[0].legend(loc="upper left")
    clean_axis(axes[0], "y")
    panel_label(axes[0], "a")

    sex_counts = patient.groupby(["cohort", "sex"]).size().unstack(fill_value=0).reindex(columns=SEX_ORDER, fill_value=0)
    x = np.arange(len(sex_counts))
    bottom = np.zeros(len(sex_counts))
    for sex in SEX_ORDER:
        values = sex_counts[sex].to_numpy()
        axes[1].bar(x, values, bottom=bottom, width=0.68, color=GROUP_COLORS[sex], label=sex, edgecolor="white", linewidth=0.4)
        bottom += values
    axes[1].set_xticks(x, sex_counts.index, rotation=25, ha="right")
    axes[1].set_ylabel("Patients")
    axes[1].legend(loc="upper right")
    clean_axis(axes[1], "y")
    panel_label(axes[1], "b")

    age_counts = patient.groupby(["cohort", "age_group"], observed=False).size().unstack(fill_value=0).reindex(columns=AGE_GROUP_ORDER, fill_value=0)
    x = np.arange(len(age_counts))
    bottom = np.zeros(len(age_counts))
    for age_group in AGE_GROUP_ORDER:
        values = age_counts[age_group].to_numpy()
        axes[2].bar(x, values, bottom=bottom, width=0.68, color=GROUP_COLORS[age_group], label=age_group, edgecolor="white", linewidth=0.4)
        bottom += values
    axes[2].set_xticks(x, age_counts.index, rotation=25, ha="right")
    axes[2].set_ylabel("Patients")
    axes[2].legend(loc="upper right")
    clean_axis(axes[2], "y")
    panel_label(axes[2], "c")
    fig.subplots_adjust(wspace=0.42)
    save_figure(fig, figure_dir / "fig_demographic_composition")


def plot_metric_forest(metrics: pd.DataFrame, figure_dir: Path) -> None:
    if metrics.empty:
        return
    tasks = [task for task in TASKS if task in set(metrics["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.55 * len(tasks), 3.2), squeeze=False)
    for panel, task in enumerate(tasks):
        ax = axes[0, panel]
        metric = PRIMARY_METRIC[task]
        subset = metrics[(metrics["task"] == task) & (metrics["metric"] == metric)].copy()
        subset["label"] = subset.apply(
            lambda row: analysis_group_label(row["cohort"], row["strategy"])
            + " | "
            + str(row["subgroup"]),
            axis=1,
        )
        subset = subset.reset_index(drop=True)
        y = np.arange(len(subset))[::-1]
        for position, (_, row) in zip(y, subset.iterrows()):
            color = GROUP_COLORS.get(str(row["subgroup"]), COLORS["black"])
            ax.errorbar(
                row["estimate"],
                position,
                xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.0,
                elinewidth=1.0,
                markeredgecolor="white",
                markeredgewidth=0.35,
                zorder=3,
            )
        ax.set_yticks(y, subset["label"])
        ax.set_xlabel(f"{METRIC_LABELS[metric]} (95% CI)")
        finite = subset[["ci_low", "ci_high"]].to_numpy(float)
        if np.isfinite(finite).any():
            fit_metric_axis(ax, subset[["estimate", "ci_low", "ci_high"]].to_numpy(float), metric)
        for collection in ax.collections:
            collection.set_linewidth(1.4)
        clean_axis(ax, "x")
        panel_label(ax, chr(ord("a") + panel))
        ax.text(0.99, 1.02, "Binary classification" if task == "bin" else "Stage classification", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
    fig.subplots_adjust(wspace=0.7)
    save_figure(fig, figure_dir / "fig_subgroup_performance_forest")


def plot_gap_forest(gaps: pd.DataFrame, figure_dir: Path) -> None:
    if gaps.empty:
        return
    selected = gaps[gaps.apply(lambda row: row["metric"] == PRIMARY_METRIC.get(row["task"]), axis=1)].copy()
    selected["label"] = selected.apply(
        lambda row: analysis_group_label(row["cohort"], row["strategy"])
        + " | "
        + str(row["comparison"])
        + " | "
        + str(row["task"]),
        axis=1,
    )
    selected = selected.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.28 * len(selected) + 0.8)))
    y = np.arange(len(selected))[::-1]
    ax.axvline(0.0, color=COLORS["black"], linestyle="--", linewidth=0.8, zorder=1)
    for position, (_, row) in zip(y, selected.iterrows()):
        color = COLORS["red"] if np.isfinite(row.get("p_holm", np.nan)) and row["p_holm"] < 0.05 else COLORS["blue"]
        ax.errorbar(
            row["difference"],
            position,
            xerr=[[row["difference"] - row["ci_low"]], [row["ci_high"] - row["difference"]]],
            fmt="o",
            color=color,
            capsize=2.0,
            elinewidth=1.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )
    ax.set_yticks(y, selected["label"])
    ax.set_xlabel("Difference in primary metric (95% CI)")
    clean_axis(ax, "x")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", label="Holm-adjusted P ≥ 0.05"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["red"], markeredgecolor="white", label="Holm-adjusted P < 0.05"),
    ]
    ax.legend(handles=legend, loc="lower right")
    save_figure(fig, figure_dir / "fig_subgroup_metric_gaps")


def plot_transfer_strategy_differences(differences: pd.DataFrame, figure_dir: Path) -> None:
    if differences.empty:
        return
    selected = differences[
        differences.apply(lambda row: row["metric"] == PRIMARY_METRIC.get(row["task"]), axis=1)
    ].copy()
    if selected.empty:
        return
    selected["label"] = (
        selected["task"].astype(str)
        + " | "
        + selected["subgroup"].astype(str)
        + " | "
        + selected["comparison"].astype(str)
    )
    selected = selected.reset_index(drop=True)
    y = np.arange(len(selected))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.25 * len(selected) + 0.9)))
    ax.axvline(0.0, color=COLORS["black"], linestyle="--", linewidth=0.8)
    for position, row in zip(y, selected.itertuples(index=False)):
        color = COLORS["red"] if np.isfinite(row.p_holm) and row.p_holm < 0.05 else COLORS["blue"]
        ax.errorbar(
            row.difference,
            position,
            xerr=[[row.difference - row.ci_low], [row.ci_high - row.difference]],
            fmt="o",
            color=color,
            capsize=2.0,
            elinewidth=1.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.set_yticks(y, selected["label"])
    ax.set_xlabel("Paired difference in primary metric (95% CI)")
    clean_axis(ax, "x")
    save_figure(fig, figure_dir / "fig_uih_transfer_strategy_differences")


def _marginal_age_curve(fit: LogisticFit, age_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean: pd.DataFrame = getattr(fit, "design_frame")
    age_center = float(getattr(fit, "age_center"))
    curves = []
    gradients = []
    for age in age_grid:
        rows = []
        for _, observation in clean.iterrows():
            row = [1.0, (age - age_center) / 10.0, float(observation["sex"] == "Male"), float(observation["side"] == "right")]
            for column in fit.columns[4:]:
                label = int(column.rsplit(" ", 1)[-1])
                row.append(float(int(observation["label"]) == label))
            rows.append(row)
        x = np.asarray(rows, dtype=float)
        probability = expit(x @ fit.beta)
        mean_probability = probability.mean()
        gradient = (x * (probability * (1.0 - probability))[:, None]).mean(axis=0)
        curves.append(mean_probability)
        gradients.append(gradient)
    curves = np.asarray(curves)
    gradients = np.asarray(gradients)
    variance = np.einsum("ij,jk,ik->i", gradients, fit.covariance, gradients)
    standard_error = np.sqrt(np.clip(variance, 0.0, None))
    return curves, np.clip(curves - 1.96 * standard_error, 0, 1), np.clip(curves + 1.96 * standard_error, 0, 1)


def plot_continuous_age_effect(
    predictions: pd.DataFrame,
    fits: Mapping[Tuple[str, str, str], LogisticFit],
    figure_dir: Path,
) -> None:
    if not fits:
        return
    tasks = [task for task in TASKS if any(key[2] == task for key in fits)]
    fig, axes = plt.subplots(
        1, len(tasks), figsize=((89 if len(tasks) == 1 else 183) / 25.4, 3.05), squeeze=False
    )
    analysis_groups = list(
        predictions[["cohort", "strategy"]]
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    strategy_colors = {
        SIEMENS_STRATEGY: COLORS["blue"],
        "zero_shot": COLORS["blue"],
        "siemens_finetune": COLORS["red"],
        "uih_ssl_only": COLORS["green"],
    }
    handles, labels = [], []
    for panel, task in enumerate(tasks):
        ax = axes[0, panel]
        penalized_present = False
        probability_bounds = []
        for cohort, strategy in analysis_groups:
            fit = fits.get((cohort, strategy, task))
            group = predictions[
                (predictions["cohort"].astype(str) == cohort)
                & (predictions["strategy"].astype(str) == strategy)
                & (predictions["task"] == task)
            ]
            if fit is None or group.empty:
                continue
            ages = group["age"].dropna()
            if ages.empty:
                continue
            low_age, high_age = np.percentile(ages, [2.5, 97.5])
            age_grid = np.linspace(low_age, high_age, 100)
            estimate, ci_low, ci_high = _marginal_age_curve(fit, age_grid)
            probability_bounds.extend((estimate, ci_low, ci_high))
            color = strategy_colors.get(strategy, COLORS["purple"])
            linestyle = "--" if fit.penalized else "-"
            curve_label = analysis_group_label(cohort, strategy)
            if fit.penalized:
                curve_label += " †"
                penalized_present = True
            line = ax.plot(
                age_grid,
                estimate,
                color=color,
                linestyle=linestyle,
                linewidth=1.65,
                label=curve_label,
                zorder=3,
            )[0]
            ax.fill_between(
                age_grid,
                ci_low,
                ci_high,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
            patient_ages = group.drop_duplicates("patient_id")["age"].dropna().to_numpy(float)
            ax.plot(
                patient_ages,
                np.full_like(patient_ages, -0.035),
                "|",
                transform=ax.get_xaxis_transform(),
                color=color,
                alpha=0.30,
                markersize=3.2,
                clip_on=False,
            )
            if curve_label not in labels:
                handles.append(line)
                labels.append(curve_label)
        for boundary in (40, 60):
            ax.axvline(
                boundary,
                color=COLORS["gray"],
                linewidth=0.75,
                linestyle=(0, (2, 2)),
                zorder=1,
            )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Adjusted error probability")
        if probability_bounds:
            from matplotlib.ticker import MaxNLocator
            finite = np.concatenate(probability_bounds)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                low, high = float(finite.min()), float(finite.max())
                span = max(high - low, .08)
                ax.set_ylim(max(0, low - .12 * span), min(1, high + .18 * span))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.grid(axis="y", color="#E8EBED", linewidth=0.45)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel))
        ax.set_title(
            "Binary classification" if task == "bin" else "Stage classification",
            loc="left",
            pad=8,
            fontweight="normal",
        )
        if penalized_present:
            ax.text(
                0.01,
                0.985,
                "† Ridge-stabilized descriptive curve",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.2,
                color=COLORS["gray"],
            )
    if handles:
        age_legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.015),
            borderaxespad=0,
        )
        age_legend.set_gid("age-effect-legend")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.24, top=0.88, wspace=0.36)
    save_figure(fig, figure_dir / "fig_continuous_age_effect")


def plot_binary_curves(predictions: pd.DataFrame, figure_dir: Path) -> None:
    data = predictions[predictions["task"] == "bin"]
    if data.empty:
        return
    for (cohort, strategy), cohort_df in data.groupby(
        ["cohort", "strategy"], sort=False
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05))
        legend_handles, legend_labels = [], []
        level_specs = [
            ("overall", "Overall"),
            *(("sex", level) for level in SEX_ORDER),
            *(("age_group", level) for level in AGE_GROUP_ORDER),
        ]
        for variable, level in level_specs:
            group = cohort_df[cohort_df[variable].astype(str) == level]
            y_true = group["label"].to_numpy(int)
            probability = group["prob_positive"].to_numpy(float)
            if len(group) == 0 or np.unique(y_true).size < 2:
                continue
            color = (
                COLORS["black"] if level == "Overall" else GROUP_COLORS[level]
            )
            style = "-" if variable in ("overall", "sex") else "--"
            width = 1.9 if level == "Overall" else 1.35
            fpr, tpr, _ = roc_curve(y_true, probability)
            precision, recall, _ = precision_recall_curve(y_true, probability)
            roc_line = axes[0].plot(
                fpr,
                tpr,
                color=color,
                linestyle=style,
                linewidth=width,
                label=f"{level} (AUROC {roc_auc_score(y_true, probability):.2f})",
                zorder=3,
            )[0]
            axes[1].plot(
                recall,
                precision,
                color=color,
                linestyle=style,
                linewidth=width,
                label=f"{level} (AUPRC {average_precision_score(y_true, probability):.2f})",
                zorder=3,
            )
            if len(group) >= 20:
                observed, predicted = calibration_curve(
                    y_true,
                    probability,
                    n_bins=min(8, max(3, len(group) // 12)),
                    strategy="quantile",
                )
                axes[2].plot(
                    predicted,
                    observed,
                    marker="o",
                    markersize=3.6,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    color=color,
                    linestyle=style,
                    linewidth=width,
                    label=level,
                    zorder=3,
                )
            legend_handles.append(roc_line)
            legend_labels.append(level)
        axes[0].plot(
            [0, 1], [0, 1], color="#9A9A9A", linestyle=(0, (3, 2)),
            linewidth=0.85, zorder=1
        )
        axes[1].axhline(
            cohort_df["label"].mean(), color="#9A9A9A",
            linestyle=(0, (3, 2)), linewidth=0.85, zorder=1
        )
        axes[2].plot(
            [0, 1], [0, 1], color="#9A9A9A", linestyle=(0, (3, 2)),
            linewidth=0.85, zorder=1
        )
        labels = (
            ("False-positive rate", "True-positive rate", "ROC curve"),
            ("Recall", "Precision", "Precision–recall curve"),
            ("Mean predicted probability", "Observed event fraction", "Calibration"),
        )
        for index, (ax, (xlabel, ylabel, title)) in enumerate(zip(axes, labels)):
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(title, loc="left", fontweight="bold", pad=7)
            ax.set_xlim(-0.015, 1.015)
            ax.set_ylim(-0.015, 1.025)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(color="#EAEAEA", linewidth=0.5, zorder=0)
            ax.set_box_aspect(1)
            clean_axis(ax)
            panel_label(ax, chr(ord("a") + index))
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(6, len(legend_handles)),
                bbox_to_anchor=(0.5, 0.015),
                frameon=False,
                handlelength=2.2,
                columnspacing=1.15,
            )
        fig.suptitle(
            analysis_group_label(cohort, strategy),
            x=0.08,
            y=0.985,
            ha="left",
            fontweight="bold",
        )
        fig.subplots_adjust(
            left=0.075, right=0.99, bottom=0.25, top=0.82, wspace=0.34
        )
        save_figure(
            fig,
            figure_dir
            / f"fig_binary_curves_{safe_name(str(cohort))}_{safe_name(str(strategy))}",
        )


def plot_stage_confusions(predictions: pd.DataFrame, figure_dir: Path) -> None:
    data = predictions[predictions["task"] == "stage"]
    if data.empty:
        return
    for (cohort, strategy), cohort_df in data.groupby(["cohort", "strategy"], sort=False):
        groups: List[Tuple[str, pd.DataFrame]] = []
        groups.extend((level, cohort_df[cohort_df["sex"].astype(str) == level]) for level in SEX_ORDER)
        groups.extend((level, cohort_df[cohort_df["age_group"].astype(str) == level]) for level in AGE_GROUP_ORDER)
        groups = [(name, group) for name, group in groups if not group.empty]
        fig, axes = plt.subplots(1, len(groups), figsize=(1.55 * len(groups), 1.75), squeeze=False)
        image = None
        for index, (name, group) in enumerate(groups):
            ax = axes[0, index]
            cm = confusion_matrix(group["label"], group["prediction"], labels=[0, 1, 2]).astype(float)
            cm /= np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
            image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            for row in range(3):
                for column in range(3):
                    ax.text(column, row, f"{cm[row, column]:.2f}", ha="center", va="center", fontsize=6.2, color="white" if cm[row, column] > 0.55 else COLORS["black"])
            ax.set_xticks(range(3), STAGE_NAMES, rotation=45, ha="right")
            ax.set_yticks(range(3), STAGE_NAMES if index == 0 else [])
            ax.set_xlabel("Predicted")
            if index == 0:
                ax.set_ylabel("True")
            ax.set_title(f"{name}\nn={group['patient_id'].nunique()}", fontsize=7.5)
            ax.tick_params(direction="out", length=0)
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02, label="Row proportion")
        fig.subplots_adjust(wspace=0.32)
        save_figure(
            fig,
            figure_dir / f"fig_stage_confusion_{safe_name(str(cohort))}_{safe_name(str(strategy))}",
        )


def plot_cross_cohort_heatmap(metrics: pd.DataFrame, figure_dir: Path) -> None:
    if metrics.empty or metrics[["cohort", "strategy"]].drop_duplicates().shape[0] < 2:
        return
    selected = metrics[metrics.apply(lambda row: row["metric"] == PRIMARY_METRIC.get(row["task"]), axis=1)].copy()
    selected["column"] = selected["task"].astype(str) + " | " + selected["subgroup"].astype(str)
    selected["analysis_group"] = selected.apply(
        lambda row: analysis_group_label(row["cohort"], row["strategy"]), axis=1
    )
    table = selected.pivot_table(index="analysis_group", columns="column", values="estimate", aggfunc="first")
    fig, ax = plt.subplots(figsize=(7.2, max(1.8, 0.42 * len(table) + 0.7)))
    values = table.to_numpy(float)
    image = ax.imshow(values, cmap="viridis", vmin=max(-0.1, float(np.nanmin(values)) - 0.05), vmax=1.0, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.7, color="white" if value < 0.45 else COLORS["black"])
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.tick_params(direction="out", length=0)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Primary metric")
    save_figure(fig, figure_dir / "fig_cross_cohort_subgroup_heatmap")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "cohort"


def resolve_task_checkpoints(args: argparse.Namespace, task: str) -> List[str]:
    explicit_values = getattr(args, f"{task}_checkpoints", [])
    checkpoint_dir = getattr(args, f"{task}_checkpoint_dir", None)
    checkpoint_name = getattr(args, f"{task}_checkpoint_name", f"{task}_full_ssl_best.pth")
    explicit_paths = expand_checkpoints(explicit_values)
    if explicit_paths and checkpoint_dir:
        raise ValueError(
            f"{task} 不应同时设置 checkpoint 路径列表和 checkpoint 文件夹，请保留一种方式。"
        )
    if checkpoint_dir:
        return discover_fold_checkpoints(Path(checkpoint_dir), checkpoint_name, task)
    return [path for _, path in validate_checkpoints(explicit_paths, task)] if explicit_paths else []


def run_inference_command(args: argparse.Namespace, external: bool) -> None:
    set_publication_style()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_predictions: List[pd.DataFrame] = []
    used_checkpoints: Dict[str, List[Dict[str, object]]] = {}
    runner = run_external_ensemble_for_task if external else run_internal_oof_for_task

    for task in TASKS:
        checkpoints = resolve_task_checkpoints(args, task)
        if not checkpoints:
            print(f"[SKIP] 未发现 {task} checkpoint，跳过该任务。")
            continue
        fold_checkpoints = validate_checkpoints(checkpoints, task)
        used_checkpoints[task] = [
            {"fold": fold, "path": path} for fold, path in fold_checkpoints
        ]
        print(f"[INFO] {task} 使用 fold：{[fold for fold, _ in fold_checkpoints]}")
        all_predictions.append(runner(task, checkpoints, args, device))
    if not all_predictions:
        raise ValueError("bin 和 stage 中至少需要发现一个可用 fold checkpoint。")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["strategy"] = "zero_shot" if external else SIEMENS_STRATEGY
    predictions, audit = attach_demographics(predictions, args.demographics_xlsx, args.complete_case)
    predictions.to_csv(output_dir / "subgroup_predictions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "demographic_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "mode": "external-ensemble" if external else "internal-oof",
        "cohort": args.cohort,
        "strategy": "zero_shot" if external else SIEMENS_STRATEGY,
        "data_root": args.data_root,
        "demographics_xlsx": args.demographics_xlsx,
        "threshold": args.threshold,
        "volume_size": list(args.volume_size),
        "complete_case": args.complete_case,
        "n_rows": len(predictions),
        "n_patients": predictions["patient_id"].nunique(),
        "tasks": sorted(predictions["task"].unique().tolist()),
        "used_checkpoints": used_checkpoints,
        "folds_by_task": {
            task: [item["fold"] for item in items] for task, items in used_checkpoints.items()
        },
        "fold_count_by_task": {
            task: len(items) for task, items in used_checkpoints.items()
        },
        "is_complete_five_fold_by_task": {
            task: len(items) == 5 and not (
                "evaluation_mode" in predictions and predictions.loc[predictions.task == task, "evaluation_mode"].eq(
                    "repeated_holdout_eligible_model_average").any()
            ) for task, items in used_checkpoints.items()
        },
        "evaluation_modes": sorted(predictions.get("evaluation_mode", pd.Series(dtype=str)).dropna().unique().tolist()),
    }
    with open(output_dir / "subgroup_inference_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[OK] 标准化预测：{output_dir / 'subgroup_predictions.csv'}")
    print(audit.to_string(index=False))


def validate_analysis_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "cohort",
        "strategy",
        "task",
        "patient_id",
        "side",
        "label",
        "prediction",
        "sex",
        "age",
        "age_group",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"合并预测表缺少字段：{sorted(missing)}")
    unknown_tasks = set(predictions["task"].dropna().astype(str)) - set(TASKS)
    if unknown_tasks:
        raise ValueError(f"未知任务：{sorted(unknown_tasks)}")
    for task, group in predictions.groupby("task"):
        prediction_columns_ok(group, str(task))


def _run_single_cohort_analysis(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    source_csvs: Sequence[str],
) -> None:
    from utils.subgroup_reporting import analyze_cohort
    return analyze_cohort(sys.modules[__name__], predictions, args, output_dir, source_csvs)



def run_analysis_command(args: argparse.Namespace) -> None:
    frames: List[pd.DataFrame] = []
    for path in args.prediction_csvs:
        frame = pd.read_csv(path)
        if "cohort" not in frame.columns:
            raise ValueError(f"预测CSV缺少cohort字段：{path}")
        frame["__source_csv"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    available_cohorts = combined["cohort"].dropna().astype(str).drop_duplicates().tolist()
    requested_cohorts = getattr(args, "cohorts", None)
    if requested_cohorts:
        cohorts = list(dict.fromkeys(map(str, requested_cohorts)))
        missing = set(cohorts) - set(available_cohorts)
        if missing:
            raise ValueError(
                f"指定队列在预测CSV中不存在：{sorted(missing)}；"
                f"可用队列：{available_cohorts}"
            )
    else:
        cohorts = available_cohorts
    if not cohorts:
        raise ValueError("给定预测CSV中没有可分析的数据队列。")

    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows: List[Dict[str, object]] = []
    for position, cohort in enumerate(cohorts, start=1):
        cohort_frame = combined[combined["cohort"].astype(str) == cohort].copy()
        source_csvs = cohort_frame.pop("__source_csv").drop_duplicates().astype(str).tolist()
        cohort_output = output_root / safe_name(cohort)
        print(f"[ANALYZE {position}/{len(cohorts)}] cohort={cohort} -> {cohort_output}")
        _run_single_cohort_analysis(cohort_frame, args, cohort_output, source_csvs)
        index_rows.append(
            {
                "cohort": cohort,
                "output_dir": str(cohort_output),
                "prediction_csvs": source_csvs,
            }
        )

    with (output_root / "cohort_analysis_index.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "analysis_mode": "separate-by-cohort",
                "cohort_order": cohorts,
                "results": index_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _build_direct_inference_args(
    config: DirectRunConfig,
    cohort: str,
    data_root: Path,
    demographics_xlsx: Path,
    out_dir: Path,
) -> argparse.Namespace:
    selected_tasks = set(config.tasks)
    return argparse.Namespace(
        cohort=cohort,
        data_root=str(data_root),
        demographics_xlsx=str(demographics_xlsx),
        bin_checkpoints=[],
        stage_checkpoints=[],
        bin_checkpoint_dir=(str(config.bin_checkpoint_dir) if "bin" in selected_tasks else None),
        bin_checkpoint_name=config.bin_checkpoint_name,
        stage_checkpoint_dir=(str(config.stage_checkpoint_dir) if "stage" in selected_tasks else None),
        stage_checkpoint_name=config.stage_checkpoint_name,
        out_dir=str(out_dir),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=config.device,
        threshold=config.threshold,
        volume_size=config.volume_size,
        complete_case=config.complete_case,
        group_by_base_id=config.group_by_base_id,
        repeated_holdout_7_3=config.repeated_holdout_7_3,
        real_only_test=config.real_only_test,
        seed=config.seed,
    )


def run_direct_workflow(config: DirectRunConfig = DIRECT_RUN_CONFIG) -> None:
    from utils.subgroup_reporting import direct_workflow
    steps = tuple(step for step, enabled in (("inference", config.run_inference), ("analysis", config.run_analyze)) if enabled)
    return direct_workflow(sys.modules[__name__], config, steps, config.cohorts)



def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cohort", help="输出中的队列名称，例如 SIEMENS_internal 或 UIH_external。",default='SIEMENS_internal')
    parser.add_argument("--data_root", help="队列的 ONFH 数据清单 JSON 路径。",default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS_Generation\50_7\siemens_onfh_data_generation.json")
    parser.add_argument("--demographics_xlsx", help="包含患者 ID、性别和年龄的人口学 Excel 路径。",default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\info_details.xlsx")
    parser.add_argument("--bin_checkpoints", nargs="*", default=[], help="显式指定 bin checkpoint，可提供 1--5 折并使用 glob。")
    parser.add_argument("--stage_checkpoints", nargs="*", default=[], help="显式指定 stage checkpoint，可提供 1--5 折并使用 glob。")
    parser.add_argument("--bin_checkpoint_dir", help="bin 权重根目录；程序会递归查找各 fold 子目录。", default=str(DIRECT_RUN_CONFIG.bin_checkpoint_dir))
    parser.add_argument("--stage_checkpoint_dir", help="stage 权重根目录；程序会递归查找各 fold 子目录。", default=str(DIRECT_RUN_CONFIG.stage_checkpoint_dir))
    parser.add_argument("--bin_checkpoint_name", default="bin_full_ssl_best.pth", help="bin 权重文件名。")
    parser.add_argument("--stage_checkpoint_name", default="stage_full_ssl_best.pth", help="stage 权重文件名。")
    parser.add_argument("--out_dir", help="预测表、审计表和运行清单的输出目录。",default='outputs/subgroups/siemens')
    parser.add_argument("--batch_size", type=int, default=2, help="推理批量大小；显存不足时设为 1。")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载并行进程数；Windows 建议设为 0。")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda 或 cpu；CUDA 不可用时自动回退 CPU。")
    parser.add_argument("--threshold", type=float, default=0.5, help="bin 二分类阳性判定阈值。")
    parser.add_argument("--volume_size", type=int, nargs=3, default=(19, 314, 224), help="输入体尺寸，顺序为层数、高度、宽度。")
    parser.add_argument("--complete_case", action=argparse.BooleanOptionalAction, default=False, help="是否仅保留性别和年龄均完整的患者。")
    parser.add_argument("--group_by_base_id", action=argparse.BooleanOptionalAction, default=False, help="是否按基础患者 ID 进行交叉验证分组。")
    parser.add_argument("--repeated_holdout_7_3", action=argparse.BooleanOptionalAction, default=False, help="是否复现训练时的 7:3 重复留出划分。")
    parser.add_argument(
        "--real_only_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否复现数据层面五折：全部合成样本进入训练，内部测试折仅包含真实样本。",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SACON 性别和年龄亚组评估（患者聚类统计与期刊风格可视化）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    workflow = subparsers.add_parser("workflow", help="两步工作流：推理/预测准备，以及统计、出图和解释。")
    workflow.add_argument("--steps", choices=("all", "inference", "analysis"), default="analysis")
    workflow.add_argument("--cohorts", nargs="+", choices=("internal", "external"), default=["internal", "external"])
    workflow.add_argument("--out_dir", default=str(DIRECT_RUN_CONFIG.output_root))
    workflow.add_argument("--external_results_root", default=str(DIRECT_RUN_CONFIG.external_results_root))
    workflow.add_argument("--n_bootstrap", type=int, default=2000)
    from utils.subgroup_reporting import workflow_command
    workflow.set_defaults(handler=lambda args: workflow_command(sys.modules[__name__], args))

    internal = subparsers.add_parser("internal-oof", help="内部队列可用折的折外推理。")
    add_inference_arguments(internal)
    internal.set_defaults(handler=lambda args: run_inference_command(args, external=False))

    external = subparsers.add_parser("external-ensemble", help="外部队列可用折的概率集成。")
    add_inference_arguments(external)
    external.set_defaults(handler=lambda args: run_inference_command(args, external=True))

    adaptation = subparsers.add_parser(
        "uih-adaptation",
        help="接入UIH迁移学习折外预测，并合并人口学信息供亚组分析。",
    )
    adaptation_source = adaptation.add_mutually_exclusive_group(required=True)
    adaptation_source.add_argument("--external_results_root", help="六个外部任务目录的根目录。")
    adaptation_source.add_argument(
        "--adaptation_prediction_csv",
        help="uih_supervised_adaptation.py生成的all_oof_predictions.csv路径。",
    )
    adaptation.add_argument(
        "--demographics_xlsx",
        required=True,
        help="UIH人口学Excel路径，需包含患者ID、性别和年龄。",
    )
    adaptation.add_argument(
        "--strategies",
        nargs="+",
        choices=UIH_TRANSFER_STRATEGIES,
        default=list(UIH_TRANSFER_STRATEGIES),
        help="需要纳入UIH亚组分析的迁移策略。",
    )
    adaptation.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="需要从UIH迁移预测中纳入的任务：bin、stage或两者。",
    )
    adaptation.add_argument(
        "--out_dir",
        required=True,
        help="标准化亚组预测表、人口学审计表和运行清单的输出目录。",
    )
    adaptation.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="UIH二分类阳性判定阈值；将根据prob_positive重新生成类别预测。",
    )
    adaptation.add_argument(
        "--complete_case",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否仅保留性别和年龄均完整的UIH患者。",
    )
    adaptation.set_defaults(handler=run_uih_adaptation_command)

    analyze = subparsers.add_parser("analyze", help="按照cohort分别对预测CSV做亚组统计与绘图。")
    analyze.add_argument("--prediction_csvs", nargs="+", required=True, help="一个或多个带人口学字段的标准化预测 CSV。")
    analyze.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="需要依次分析的队列名称及顺序；不指定时按CSV中首次出现顺序分析全部队列。",
    )
    analyze.add_argument("--out_dir", required=True, help="统计表和期刊风格图形的输出目录。")
    analyze.add_argument("--n_bootstrap", type=int, default=2000, help="患者聚类 bootstrap 重复次数。")
    analyze.add_argument("--seed", type=int, default=2026, help="bootstrap 等随机过程的随机种子。")
    analyze.set_defaults(handler=run_analysis_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    command_line = list(sys.argv[1:] if argv is None else argv)
    if command_line:
        args = build_parser().parse_args(command_line)
        seed = getattr(args, "seed", 2026)
        handler: Callable[[], None] = lambda: args.handler(args)
    else:
        seed = DIRECT_RUN_CONFIG.seed
        handler = lambda: run_direct_workflow(DIRECT_RUN_CONFIG)

    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            handler()
    except FileNotFoundError as exc:
        if not command_line:
            raise SystemExit(f"[ERROR] {exc}") from None
        raise


if __name__ == "__main__":
    main()
