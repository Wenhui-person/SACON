
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, mean_absolute_error,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm

from utils.publication_style import (
    COLORS, STRATEGY_COLORS, STRATEGY_LINESTYLES, STRATEGY_MARKERS,
    add_panel_labels, audit_figure, clean_axis, create_nature_figure,
    save_publication_figure, save_source_data,
)

TASKS = ("bin", "stage")
STRATEGIES = ("zero_shot", "siemens_finetune", "uih_ssl_only")
STRATEGY_LABELS = {
    "zero_shot": "Zero-shot",
    "siemens_finetune": "SIEMENS fine-tune",
    "uih_ssl_only": "UIH SSL-only",
}
PRIMARY_METRIC = {"bin": "auroc", "stage": "qwk"}
RESULT_LAYOUT = {
    ("bin", "zero_shot"): ("zero_shot-uih-bin", "bin", "bin_zero_shot_ensemble_predictions.csv"),
    ("bin", "siemens_finetune"): ("siemens_finetune-bin", "bin", "bin_oof_predictions.csv"),
    ("bin", "uih_ssl_only"): ("ssl-uih-bin", "bin", "bin_oof_predictions.csv"),
    ("stage", "zero_shot"): ("zero_shot-uih-stage", "stage", "stage_zero_shot_ensemble_predictions.csv"),
    ("stage", "siemens_finetune"): ("siemens_finetune-stage", "stage", "stage_oof_predictions.csv"),
    ("stage", "uih_ssl_only"): ("ssl-uih-stage", "stage", "stage_oof_predictions.csv"),
}
CONTRASTS = (
    ("siemens_finetune", "zero_shot"),
    ("uih_ssl_only", "zero_shot"),
    ("uih_ssl_only", "siemens_finetune"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_columns(task: str) -> set[str]:
    common = {"patient_id", "side", "label", "prediction"}
    return common | ({"prob_positive"} if task == "bin" else {"prob_class_0", "prob_class_1", "prob_class_2"})


def _validate_prediction_frame(frame: pd.DataFrame, task: str, strategy: str, source: Path) -> dict:
    missing = _required_columns(task) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing required columns: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError(f"{source} contains duplicate columns.")
    keys = ["patient_id", "side"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{source} contains duplicate patient_id-side predictions.")
    if not set(frame["side"].astype(str)).issubset({"left", "right"}):
        raise ValueError(f"{source} contains an unknown side.")
    labels = frame["label"].astype(int).to_numpy()
    predictions = frame["prediction"].astype(int).to_numpy()
    allowed = {0, 1} if task == "bin" else {0, 1, 2}
    if not set(labels).issubset(allowed) or not set(predictions).issubset(allowed):
        raise ValueError(f"{source} contains labels/predictions outside {sorted(allowed)}.")
    if task == "bin":
        probability = frame["prob_positive"].astype(float).to_numpy()
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{source} contains invalid binary probabilities.")
        expected = (probability >= 0.5).astype(int)
    else:
        probability = frame[[f"prob_class_{index}" for index in range(3)]].astype(float).to_numpy()
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{source} contains invalid stage probabilities.")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{source} stage probabilities do not sum to one.")
        expected = probability.argmax(axis=1)
    if not np.array_equal(expected, predictions):
        raise ValueError(f"{source} predictions are inconsistent with locked threshold/argmax.")
    return {
        "task": task,
        "strategy": strategy,
        "evaluation_design": "external_zero_shot_ensemble" if strategy == "zero_shot" else "target_domain_oof_adaptation",
        "source_file": str(source),
        "sha256": _sha256(source),
        "n_rows": int(len(frame)),
        "n_patients": int(frame["patient_id"].astype(str).nunique()),
        "n_left": int((frame["side"].astype(str) == "left").sum()),
        "n_right": int((frame["side"].astype(str) == "right").sum()),
    }


def load_analysis_predictions(root: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    frames, audit_rows = [], []
    reference: Dict[str, pd.DataFrame] = {}
    for task in TASKS:
        for strategy in STRATEGIES:
            directory, task_dir, filename = RESULT_LAYOUT[(task, strategy)]
            path = root / directory / task_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Expected formal prediction file not found: {path}")
            frame = pd.read_csv(path)
            audit_rows.append(_validate_prediction_frame(frame, task, strategy, path))
            frame = frame.copy()
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["side"] = frame["side"].astype(str)
            frame["task"] = task
            frame["strategy"] = strategy
            frame["evaluation_design"] = (
                "external_zero_shot_ensemble" if strategy == "zero_shot" else "target_domain_oof_adaptation"
            )
            keep = ["task", "strategy", "evaluation_design", "patient_id", "side", "label", "prediction"]
            keep += ["prob_positive"] if task == "bin" else [f"prob_class_{index}" for index in range(3)]
            frame = frame[keep].sort_values(["patient_id", "side"]).reset_index(drop=True)
            if task not in reference:
                reference[task] = frame[["patient_id", "side", "label"]].copy()
            elif not reference[task].equals(frame[["patient_id", "side", "label"]]):
                raise ValueError(f"{task}/{strategy} does not match the reference patient-side labels.")
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    bin_ref = reference["bin"].rename(columns={"label": "bin_label"})
    stage_ref = reference["stage"].rename(columns={"label": "stage_label"})
    cross = bin_ref.merge(stage_ref, on=["patient_id", "side"], validate="one_to_one")
    cross["expected_bin_label_from_stage"] = (cross["stage_label"].to_numpy(int) > 0).astype(int)
    cross["label_consistent"] = cross["bin_label"].astype(int) == cross["expected_bin_label_from_stage"]
    mismatch = cross[~cross["label_consistent"]]
    audit = pd.DataFrame(audit_rows)
    audit["cross_task_label_consistency"] = bool(mismatch.empty)
    audit["cross_task_label_mismatch_count"] = int(len(mismatch))
    audit["cross_task_label_mismatch_keys"] = ";".join(
        f"{row.patient_id}|{row.side}" for row in mismatch.itertuples(index=False)
    )
    return combined, audit


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    data = pd.DataFrame({"label": y_true.astype(int), "probability": y_prob.astype(float)})
    bins = max(1, min(n_bins, int(data["probability"].nunique()), len(data)))
    data["bin"] = pd.qcut(data["probability"], q=bins, labels=False, duplicates="drop")
    result = data.groupby("bin", observed=True).agg(
        mean_predicted=("probability", "mean"),
        observed_rate=("label", "mean"),
        count=("label", "size"),
    ).reset_index()
    result["weight"] = result["count"] / len(data)
    return result


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = calibration_bins(y_true, y_prob, n_bins)
    return float((bins["weight"] * (bins["mean_predicted"] - bins["observed_rate"]).abs()).sum())


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    probability = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    x = np.column_stack([np.ones(len(probability)), np.log(probability / (1 - probability))])
    y = np.asarray(y_true, dtype=float)
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        eta = np.clip(x @ beta, -30, 30)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1 - fitted), 1e-8, None)
        hessian = x.T @ (weights[:, None] * x) + np.eye(2) * 1e-8
        step = np.linalg.solve(hessian, x.T @ (y - fitted))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def binary_metric_bundle(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["label"].to_numpy(int)
    y_prob = frame["prob_positive"].to_numpy(float)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)) if np.unique(y_true).size > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def stage_metric_bundle(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["label"].to_numpy(int)
    y_pred = frame["prediction"].to_numpy(int)
    probs = frame[[f"prob_class_{index}" for index in range(3)]].to_numpy(float)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    one_hot = np.eye(3)[y_true]
    abs_error = np.abs(y_pred - y_true)
    class_auc, class_ap = [], []
    for index in range(3):
        target = (y_true == index).astype(int)
        if np.unique(target).size > 1:
            class_auc.append(roc_auc_score(target, probs[:, index]))
            class_ap.append(average_precision_score(target, probs[:, index]))
    cumulative_true = np.cumsum(one_hot, axis=1)[:, :-1]
    cumulative_prob = np.cumsum(probs, axis=1)[:, :-1]
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2], weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "macro_ovr_auroc": float(np.mean(class_auc)) if class_auc else np.nan,
        "map": float(np.mean(class_ap)) if class_ap else np.nan,
        "within_one_stage_accuracy": float((abs_error <= 1).mean()),
        "severe_cross_stage_error_rate": float((abs_error > 1).mean()),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "ranked_probability_score": float(np.mean(np.sum((cumulative_prob - cumulative_true) ** 2, axis=1) / 2)),
    }


def metric_bundle(frame: pd.DataFrame, task: str) -> Dict[str, float]:
    return binary_metric_bundle(frame) if task == "bin" else stage_metric_bundle(frame)


def _patient_groups(frame: pd.DataFrame) -> Tuple[np.ndarray, Mapping[str, pd.DataFrame]]:
    patients = frame["patient_id"].astype(str).drop_duplicates().to_numpy()
    groups = {str(patient): group for patient, group in frame.groupby(frame["patient_id"].astype(str), sort=False)}
    return patients, groups


def _resample(groups: Mapping[str, pd.DataFrame], sampled: Sequence[str]) -> pd.DataFrame:
    return pd.concat([groups[str(patient)] for patient in sampled], ignore_index=True)


def bootstrap_metrics(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, distribution_rows = [], []
    for task in TASKS:
        task_data = predictions[predictions["task"] == task]
        strategy_frames = {strategy: task_data[task_data["strategy"] == strategy] for strategy in STRATEGIES}
        patients, _ = _patient_groups(strategy_frames["zero_shot"])
        grouped = {strategy: _patient_groups(frame)[1] for strategy, frame in strategy_frames.items()}
        point = {strategy: metric_bundle(frame, task) for strategy, frame in strategy_frames.items()}
        samples = {strategy: {metric: [] for metric in point[strategy]} for strategy in STRATEGIES}
        rng = np.random.default_rng(seed + (0 if task == "bin" else 10000))
        bootstrap_iter = tqdm(
            range(n_bootstrap),
            desc=f"患者聚类 bootstrap（{task}）",
            unit="次",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for replicate in bootstrap_iter:
            sampled = rng.choice(patients, size=len(patients), replace=True)
            for strategy in STRATEGIES:
                values = metric_bundle(_resample(grouped[strategy], sampled), task)
                for metric, value in values.items():
                    if np.isfinite(value):
                        samples[strategy][metric].append(float(value))
                        distribution_rows.append(
                            {
                                "task": task,
                                "strategy": strategy,
                                "metric": metric,
                                "replicate": replicate,
                                "value": float(value),
                            }
                        )
        for strategy in STRATEGIES:
            group = strategy_frames[strategy]
            for metric, estimate in point[strategy].items():
                values = np.asarray(samples[strategy][metric], dtype=float)
                summary_rows.append(
                    {
                        "task": task,
                        "strategy": strategy,
                        "metric": metric,
                        "estimate": float(estimate),
                        "ci_low": float(np.percentile(values, 2.5)) if values.size else np.nan,
                        "ci_high": float(np.percentile(values, 97.5)) if values.size else np.nan,
                        "iqr_low": float(np.percentile(values, 25)) if values.size else np.nan,
                        "iqr_high": float(np.percentile(values, 75)) if values.size else np.nan,
                        "n_patients": int(group["patient_id"].nunique()),
                        "n_hips": int(len(group)),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(order) - reverse_rank + 1
        running = min(running, values[index] * len(order) / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_strategy_comparisons(
    predictions: pd.DataFrame,
    bootstrap_distribution: pd.DataFrame,
    n_permutations: int,
    seed: int,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, difference_samples = [], []
    for task in TASKS:
        task_data = predictions[predictions["task"] == task]
        frames = {strategy: task_data[task_data["strategy"] == strategy].copy() for strategy in STRATEGIES}
        patients = frames["zero_shot"]["patient_id"].astype(str).drop_duplicates().to_numpy()
        point = {strategy: metric_bundle(frame, task) for strategy, frame in frames.items()}
        dist = bootstrap_distribution[bootstrap_distribution["task"] == task]
        rng = np.random.default_rng(seed + (50000 if task == "stage" else 40000))
        for left, right in CONTRASTS:
            left_dist = dist[dist["strategy"] == left]
            right_dist = dist[dist["strategy"] == right]
            metrics = list(point[left])
            paired = left_dist.merge(
                right_dist,
                on=["task", "metric", "replicate"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            paired["difference"] = paired["value_left"] - paired["value_right"]
            paired["left_strategy"] = left
            paired["right_strategy"] = right
            difference_samples.append(
                paired[["task", "left_strategy", "right_strategy", "metric", "replicate", "difference"]]
            )
            observed = {metric: point[left][metric] - point[right][metric] for metric in metrics}
            exceed = {metric: 0 for metric in metrics}
            valid = {metric: 0 for metric in metrics}
            left_frame = frames[left].sort_values(["patient_id", "side"]).reset_index(drop=True)
            right_frame = frames[right].sort_values(["patient_id", "side"]).reset_index(drop=True)
            if not left_frame[["patient_id", "side", "label"]].equals(
                right_frame[["patient_id", "side", "label"]]
            ):
                raise ValueError(f"Unpaired predictions for {task}: {left} versus {right}.")
            swappable = ["prediction", "prob_positive"] if task == "bin" else [
                "prediction", "prob_class_0", "prob_class_1", "prob_class_2"
            ]
            row_patients = left_frame["patient_id"].astype(str).to_numpy()
            permutation_iter = tqdm(
                range(n_permutations),
                desc=f"配对置换（{task}: {left} - {right}）",
                unit="次",
                dynamic_ncols=True,
                leave=True,
                disable=not show_progress,
            )
            for _ in permutation_iter:
                swap = rng.random(len(patients)) < 0.5
                swap_lookup = dict(zip(patients, swap))
                row_swap = np.fromiter((swap_lookup[value] for value in row_patients), dtype=bool)
                perm_left, perm_right = left_frame.copy(), right_frame.copy()
                for column in swappable:
                    left_values = left_frame[column].to_numpy()
                    right_values = right_frame[column].to_numpy()
                    perm_left[column] = np.where(row_swap, right_values, left_values)
                    perm_right[column] = np.where(row_swap, left_values, right_values)
                left_values = metric_bundle(perm_left, task)
                right_values = metric_bundle(perm_right, task)
                for metric in metrics:
                    value = left_values[metric] - right_values[metric]
                    if np.isfinite(value):
                        valid[metric] += 1
                        exceed[metric] += int(abs(value) >= abs(observed[metric]) - 1e-12)
            for metric in metrics:
                values = paired.loc[paired["metric"] == metric, "difference"].to_numpy(float)
                rows.append(
                    {
                        "task": task,
                        "left_strategy": left,
                        "right_strategy": right,
                        "comparison": f"{left} - {right}",
                        "metric": metric,
                        "difference": float(observed[metric]),
                        "ci_low": float(np.percentile(values, 2.5)) if values.size else np.nan,
                        "ci_high": float(np.percentile(values, 97.5)) if values.size else np.nan,
                        "p_value": float((exceed[metric] + 1) / (valid[metric] + 1)) if valid[metric] else np.nan,
                        "analysis_family": "primary" if metric == PRIMARY_METRIC[task] else "secondary_exploratory",
                    }
                )
    result = pd.DataFrame(rows)
    result["p_adjusted"] = np.nan
    result["adjustment"] = ""
    for task in TASKS:
        primary = (result["task"] == task) & (result["analysis_family"] == "primary")
        secondary = (result["task"] == task) & (result["analysis_family"] == "secondary_exploratory")
        result.loc[primary, "p_adjusted"] = holm_adjust(result.loc[primary, "p_value"])
        result.loc[primary, "adjustment"] = "holm_within_task_primary"
        result.loc[secondary, "p_adjusted"] = benjamini_hochberg(result.loc[secondary, "p_value"])
        result.loc[secondary, "adjustment"] = "benjamini_hochberg_within_task_secondary"
    return result, pd.concat(difference_samples, ignore_index=True)


def classwise_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, strategy), frame in predictions.groupby(["task", "strategy"], sort=False):
        y_true, y_pred = frame["label"].to_numpy(int), frame["prediction"].to_numpy(int)
        labels = [0, 1] if task == "bin" else [0, 1, 2]
        probs = (
            np.column_stack([1 - frame["prob_positive"].to_numpy(float), frame["prob_positive"].to_numpy(float)])
            if task == "bin"
            else frame[[f"prob_class_{index}" for index in range(3)]].to_numpy(float)
        )
        for class_index in labels:
            target = (y_true == class_index).astype(int)
            rows.append(
                {
                    "task": task,
                    "strategy": strategy,
                    "class_index": class_index,
                    "support": int(target.sum()),
                    "precision": float(precision_score(y_true, y_pred, labels=[class_index], average="macro", zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, labels=[class_index], average="macro", zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, labels=[class_index], average="macro", zero_division=0)),
                    "ovr_auroc": float(roc_auc_score(target, probs[:, class_index])) if np.unique(target).size > 1 else np.nan,
                    "average_precision": float(average_precision_score(target, probs[:, class_index])) if np.unique(target).size > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def side_metrics(
    predictions: pd.DataFrame,
    n_bootstrap: int = 5000,
    seed: int = 2026,
    show_progress: bool = False,
) -> pd.DataFrame:
    rows = []
    for (task, strategy, side), frame in predictions.groupby(["task", "strategy", "side"], sort=False):
        point = metric_bundle(frame, task)
        patients, groups = _patient_groups(frame)
        samples = {metric: [] for metric in point}
        rng = np.random.default_rng(seed + (0 if side == "left" else 1000) + (0 if task == "bin" else 2000))
        side_iter = tqdm(
            range(n_bootstrap),
            desc=f"侧别 bootstrap（{task}/{strategy}/{side}）",
            unit="次",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for _ in side_iter:
            sampled = rng.choice(patients, size=len(patients), replace=True)
            values = metric_bundle(_resample(groups, sampled), task)
            for metric, value in values.items():
                if np.isfinite(value):
                    samples[metric].append(value)
        for metric, value in point.items():
            distribution = np.asarray(samples[metric], dtype=float)
            rows.append(
                {
                    "task": task,
                    "strategy": strategy,
                    "side": side,
                    "metric": metric,
                    "estimate": value,
                    "ci_low": float(np.percentile(distribution, 2.5)) if distribution.size else np.nan,
                    "ci_high": float(np.percentile(distribution, 97.5)) if distribution.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def stage_calibration_table(predictions: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    rows = []
    data = predictions[predictions["task"] == "stage"]
    for strategy in STRATEGIES:
        frame = data[data["strategy"] == strategy]
        labels = frame["label"].to_numpy(int)
        for class_index in range(3):
            target = (labels == class_index).astype(int)
            curve = calibration_bins(target, frame[f"prob_class_{class_index}"].to_numpy(float), n_bins)
            curve["strategy"] = strategy
            curve["class_index"] = class_index
            rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def error_overlap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        data = predictions[predictions["task"] == task]
        wide = data.pivot(index=["patient_id", "side"], columns="strategy", values="prediction")
        labels = data.drop_duplicates(["patient_id", "side"]).set_index(["patient_id", "side"])["label"]
        correct = wide.eq(labels, axis=0)
        patterns = correct.apply(lambda row: "".join("1" if row[strategy] else "0" for strategy in STRATEGIES), axis=1)
        for pattern, count in patterns.value_counts().sort_index().items():
            rows.append({"task": task, "correctness_pattern": pattern, "count": int(count)})
    return pd.DataFrame(rows)


def stage_error_transitions(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions[predictions["task"] == "stage"].copy()
    data["error_type"] = np.select(
        [data["prediction"] == data["label"], (data["prediction"] - data["label"]).abs() == 1],
        ["correct", "adjacent_error"],
        default="severe_error",
    )
    wide = data.pivot(index=["patient_id", "side"], columns="strategy", values="error_type")
    rows = []
    for target in ("siemens_finetune", "uih_ssl_only"):
        counts = wide.groupby(["zero_shot", target]).size().reset_index(name="count")
        counts["target_strategy"] = target
        counts = counts.rename(columns={"zero_shot": "source_error_type", target: "target_error_type"})
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def cross_task_consistency(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, detail = [], []
    for strategy in STRATEGIES:
        binary = predictions[(predictions["task"] == "bin") & (predictions["strategy"] == strategy)][
            ["patient_id", "side", "label", "prob_positive", "prediction"]
        ].rename(columns={"label": "bin_label", "prediction": "bin_prediction"})
        stage = predictions[(predictions["task"] == "stage") & (predictions["strategy"] == strategy)][
            ["patient_id", "side", "label", "prob_class_0", "prediction"]
        ].rename(columns={"label": "stage_label", "prediction": "stage_prediction"})
        merged = binary.merge(stage, on=["patient_id", "side"], validate="one_to_one")
        merged["ground_truth_coherent"] = merged["bin_label"].astype(int) == (
            merged["stage_label"].astype(int) > 0
        ).astype(int)
        merged["stage_derived_onfh_probability"] = 1 - merged["prob_class_0"]
        merged["stage_derived_bin_prediction"] = (merged["stage_derived_onfh_probability"] >= 0.5).astype(int)
        merged["probability_difference"] = merged["prob_positive"] - merged["stage_derived_onfh_probability"]
        merged["strategy"] = strategy
        merged["discordance_type"] = np.select(
            [
                (merged["bin_prediction"] == 1) & (merged["stage_derived_bin_prediction"] == 0),
                (merged["bin_prediction"] == 0) & (merged["stage_derived_bin_prediction"] == 1),
            ],
            ["binary_positive_stage_normal", "binary_negative_stage_onfh"],
            default="concordant",
        )
        rows.append(
            {
                "strategy": strategy,
                "pearson_probability_correlation": float(
                    np.corrcoef(merged["prob_positive"], merged["stage_derived_onfh_probability"])[0, 1]
                ),
                "mean_absolute_probability_difference": float(merged["probability_difference"].abs().mean()),
                "mean_probability_difference": float(merged["probability_difference"].mean()),
                "decision_agreement": float(
                    (merged["bin_prediction"] == merged["stage_derived_bin_prediction"]).mean()
                ),
                "cohen_kappa": float(
                    cohen_kappa_score(merged["bin_prediction"], merged["stage_derived_bin_prediction"])
                ),
                "binary_positive_stage_normal": int(
                    (merged["discordance_type"] == "binary_positive_stage_normal").sum()
                ),
                "binary_negative_stage_onfh": int(
                    (merged["discordance_type"] == "binary_negative_stage_onfh").sum()
                ),
                "ground_truth_mismatch_count": int((~merged["ground_truth_coherent"]).sum()),
            }
        )
        detail.append(merged)
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def _density(values: np.ndarray, points: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.std(values) < 1e-12:
        center = float(values[0]) if values.size else 0.0
        return np.array([center - 1e-4, center, center + 1e-4]), np.array([0.0, 1.0, 0.0])
    std = float(np.std(values, ddof=1))
    bandwidth = max(1.06 * std * values.size ** (-0.2), 1e-5)
    grid = np.linspace(values.min() - 2 * bandwidth, values.max() + 2 * bandwidth, points)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled ** 2).mean(axis=1) / (bandwidth * math.sqrt(2 * math.pi))
    return grid, density


def plot_primary_half_eye(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    figure_dir: Path,
    source_dir: Path,
) -> None:
    fig, axes = create_nature_figure(width="double", height_mm=78, ncols=2)
    source_rows = []
    for panel, task in enumerate(TASKS):
        ax = axes[panel]
        metric = PRIMARY_METRIC[task]
        sub_summary = summary[(summary["task"] == task) & (summary["metric"] == metric)]
        sub_dist = distribution[(distribution["task"] == task) & (distribution["metric"] == metric)]
        for position, strategy in enumerate(STRATEGIES):
            row = sub_summary[sub_summary["strategy"] == strategy].iloc[0]
            values = sub_dist[sub_dist["strategy"] == strategy]["value"].to_numpy(float)
            grid, density = _density(values)
            density = density / max(density.max(), 1e-12) * 0.32
            color = STRATEGY_COLORS[strategy]
            ax.fill_between(grid, position, position + density, color=color, alpha=0.28, linewidth=0)
            ax.plot(grid, position + density, color=color, lw=0.8)
            ax.plot([row.ci_low, row.ci_high], [position, position], color=color, lw=0.8)
            ax.plot([row.iqr_low, row.iqr_high], [position, position], color=color, lw=3.0, solid_capstyle="round")
            ax.scatter(
                row.estimate, position, s=24, marker=STRATEGY_MARKERS[strategy],
                color=color, edgecolor="white", linewidth=0.45, zorder=5,
            )
            source_rows.extend(
                {"task": task, "strategy": strategy, "metric": metric, "replicate": index, "value": value}
                for index, value in enumerate(values)
            )
        ax.set_yticks(range(3), [STRATEGY_LABELS[value] for value in STRATEGIES])
        ax.set_xlabel("AUROC" if task == "bin" else "Quadratic weighted kappa")
        ax.set_title("Binary classification" if task == "bin" else "Stage classification")
        ax.invert_yaxis()
        clean_axis(ax, grid=True)
    add_panel_labels(axes)
    fig.text(0.5, 0.005, "Half-density: patient-cluster bootstrap uncertainty; point: observed estimate", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_publication_figure(fig, figure_dir / "fig1_primary_half_eye")
    plt.close(fig)
    save_source_data(pd.DataFrame(source_rows), source_dir / "fig1_primary_half_eye.csv")


def plot_gardner_altman(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    figure_dir: Path,
    source_dir: Path,
) -> None:
    fig, axes = create_nature_figure(width="double", height_mm=105, nrows=2, ncols=2)
    for row_index, task in enumerate(TASKS):
        metric = PRIMARY_METRIC[task]
        estimates = summary[(summary["task"] == task) & (summary["metric"] == metric)]
        differences = comparisons[(comparisons["task"] == task) & (comparisons["metric"] == metric)]
        left_ax, right_ax = axes[row_index]
        for position, strategy in enumerate(STRATEGIES):
            item = estimates[estimates["strategy"] == strategy].iloc[0]
            left_ax.errorbar(
                item.estimate, position,
                xerr=[[max(0.0, item.estimate - item.ci_low)], [max(0.0, item.ci_high - item.estimate)]],
                fmt=STRATEGY_MARKERS[strategy], color=STRATEGY_COLORS[strategy], capsize=2,
            )
        left_ax.set_yticks(range(3), [STRATEGY_LABELS[value] for value in STRATEGIES])
        left_ax.invert_yaxis()
        left_ax.set_xlabel(metric.upper())
        left_ax.set_title(("Binary" if task == "bin" else "Stage") + " estimates")
        clean_axis(left_ax, grid=True)
        for position, item in enumerate(differences.itertuples(index=False)):
            right_ax.errorbar(
                item.difference, position,
                xerr=[[max(0.0, item.difference - item.ci_low)], [max(0.0, item.ci_high - item.difference)]],
                fmt="o", color=COLORS["purple"], capsize=2,
            )
            right_ax.text(item.ci_high, position + 0.17, f"q={item.p_adjusted:.3g}", fontsize=6, ha="right")
        right_ax.axvline(0, color=COLORS["gray"], lw=0.7, ls=":")
        right_ax.set_yticks(
            range(len(differences)),
            [f"{STRATEGY_LABELS[item.left_strategy]} − {STRATEGY_LABELS[item.right_strategy]}" for item in differences.itertuples()],
        )
        right_ax.invert_yaxis()
        right_ax.set_xlabel("Paired difference (95% CI)")
        right_ax.set_title("Gardner–Altman contrasts")
        clean_axis(right_ax, grid=True)
    add_panel_labels(axes.ravel())
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "fig2_gardner_altman")
    plt.close(fig)
    source = pd.concat([
        summary[summary.apply(lambda row: row["metric"] == PRIMARY_METRIC[row["task"]], axis=1)].assign(source="estimate"),
        comparisons[comparisons.apply(lambda row: row["metric"] == PRIMARY_METRIC[row["task"]], axis=1)].assign(source="difference"),
    ], ignore_index=True, sort=False)
    save_source_data(source, source_dir / "fig2_gardner_altman.csv")


def plot_binary_diagnostics(predictions: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    data = predictions[predictions["task"] == "bin"]
    fig, axes = create_nature_figure(width="double", height_mm=74, ncols=3)
    source_rows = []
    prevalence = float(data.drop_duplicates(["patient_id", "side"])["label"].mean())
    for strategy in STRATEGIES:
        frame = data[data["strategy"] == strategy]
        y_true = frame["label"].to_numpy(int)
        y_prob = frame["prob_positive"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        cal = calibration_bins(y_true, y_prob, 10)
        color, line = STRATEGY_COLORS[strategy], STRATEGY_LINESTYLES[strategy]
        axes[0].plot(fpr, tpr, color=color, ls=line, label=f"{STRATEGY_LABELS[strategy]} ({roc_auc_score(y_true, y_prob):.3f})")
        axes[1].plot(recall, precision, color=color, ls=line, label=f"{STRATEGY_LABELS[strategy]} ({average_precision_score(y_true, y_prob):.3f})")
        axes[2].plot(cal["mean_predicted"], cal["observed_rate"], color=color, ls=line, marker=STRATEGY_MARKERS[strategy], label=STRATEGY_LABELS[strategy])
        for x, y in zip(fpr, tpr):
            source_rows.append({"panel": "roc", "strategy": strategy, "x": x, "y": y, "count": np.nan})
        for x, y in zip(recall, precision):
            source_rows.append({"panel": "precision_recall", "strategy": strategy, "x": x, "y": y, "count": np.nan})
        for item in cal.itertuples(index=False):
            source_rows.append({"panel": "calibration", "strategy": strategy, "x": item.mean_predicted, "y": item.observed_rate, "count": item.count})
        inset = axes[2].inset_axes([0.08, 0.02, 0.88, 0.18])
        inset.hist(y_prob, bins=np.linspace(0, 1, 16), color=color, alpha=0.24, histtype="stepfilled")
        inset.set_xlim(0, 1)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(False)
    axes[0].plot([0, 1], [0, 1], color=COLORS["gray"], ls=":", lw=0.7)
    axes[1].axhline(prevalence, color=COLORS["gray"], ls=":", lw=0.7)
    axes[2].plot([0, 1], [0, 1], color=COLORS["gray"], ls=":", lw=0.7)
    titles = ("ROC", "Precision–recall", "Calibration")
    labels = (("False-positive rate", "True-positive rate"), ("Recall", "Precision"), ("Mean predicted probability", "Observed proportion"))
    for ax, title, (xlabel, ylabel) in zip(axes, titles, labels):
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel, title=title)
        clean_axis(ax)
    axes[0].legend(fontsize=6, loc="lower right")
    axes[1].legend(fontsize=6, loc="lower left")
    axes[2].legend(fontsize=6, loc="upper left")
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "fig3_binary_diagnostics")
    plt.close(fig)
    save_source_data(pd.DataFrame(source_rows), source_dir / "fig3_binary_diagnostics.csv")


def plot_stage_bubble_confusions(predictions: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    data = predictions[predictions["task"] == "stage"]
    fig, axes = create_nature_figure(width="double", height_mm=66, ncols=3)
    source_rows = []
    names = ["Normal", "Early", "Late"]
    for ax, strategy in zip(axes, STRATEGIES):
        frame = data[data["strategy"] == strategy]
        counts = confusion_matrix(frame["label"], frame["prediction"], labels=[0, 1, 2])
        proportions = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
        for true_index in range(3):
            for predicted_index in range(3):
                value = proportions[true_index, predicted_index]
                ax.scatter(predicted_index, true_index, s=900 * value + 8, color=STRATEGY_COLORS[strategy], alpha=0.18 + 0.65 * value, edgecolor="white", linewidth=0.5)
                ax.text(predicted_index, true_index, f"{value:.2f}\n({counts[true_index, predicted_index]})", ha="center", va="center", fontsize=6)
                source_rows.append({"strategy": strategy, "true_class": true_index, "predicted_class": predicted_index, "count": counts[true_index, predicted_index], "row_proportion": value})
        ax.set_xticks(range(3), names, rotation=30, ha="right")
        ax.set_yticks(range(3), names)
        ax.set(xlim=(-0.55, 2.55), ylim=(2.55, -0.55), xlabel="Predicted", title=STRATEGY_LABELS[strategy])
        ax.set_aspect("equal")
        ax.grid(color=COLORS["light_gray"], linewidth=0.45)
    axes[0].set_ylabel("True")
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "fig4_stage_bubble_confusions")
    plt.close(fig)
    save_source_data(pd.DataFrame(source_rows), source_dir / "fig4_stage_bubble_confusions.csv")


def _draw_alluvial_panel(ax, transitions: pd.DataFrame, target: str) -> None:
    order = ["correct", "adjacent_error", "severe_error"]
    colors = {"correct": COLORS["green"], "adjacent_error": COLORS["orange"], "severe_error": COLORS["vermillion"]}
    subset = transitions[transitions["target_strategy"] == target]
    total = max(int(subset["count"].sum()), 1)
    left_totals = subset.groupby("source_error_type")["count"].sum().reindex(order, fill_value=0)
    right_totals = subset.groupby("target_error_type")["count"].sum().reindex(order, fill_value=0)
    gap = 0.035
    left_cursor = right_cursor = 0.0
    left_pos, right_pos = {}, {}
    for label in order:
        height = left_totals[label] / total * (1 - gap * 2)
        left_pos[label] = [left_cursor, left_cursor + height]
        ax.add_patch(Rectangle((0, left_cursor), 0.05, height, color=colors[label], alpha=0.85, lw=0))
        left_cursor += height + gap
        height = right_totals[label] / total * (1 - gap * 2)
        right_pos[label] = [right_cursor, right_cursor + height]
        ax.add_patch(Rectangle((0.95, right_cursor), 0.05, height, color=colors[label], alpha=0.85, lw=0))
        right_cursor += height + gap
    left_used = {label: left_pos[label][0] for label in order}
    right_used = {label: right_pos[label][0] for label in order}
    for source in order:
        for destination in order:
            match = subset[(subset["source_error_type"] == source) & (subset["target_error_type"] == destination)]
            count = int(match["count"].sum())
            if not count:
                continue
            height = count / total * (1 - gap * 2)
            y0, y1 = left_used[source], right_used[destination]
            vertices = [(0.05, y0), (0.36, y0), (0.64, y1), (0.95, y1), (0.95, y1 + height), (0.64, y1 + height), (0.36, y0 + height), (0.05, y0 + height), (0.05, y0)]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=colors[source], edgecolor="none", alpha=0.24))
            left_used[source] += height
            right_used[destination] += height
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xticks=[0.025, 0.975], xticklabels=["Zero-shot", STRATEGY_LABELS[target]], yticks=[])
    ax.set_title(f"Zero-shot → {STRATEGY_LABELS[target]}")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_stage_alluvial(transitions: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    fig, axes = create_nature_figure(width="double", height_mm=72, ncols=2)
    for ax, target in zip(axes, ("siemens_finetune", "uih_ssl_only")):
        _draw_alluvial_panel(ax, transitions, target)
    add_panel_labels(axes)
    fig.text(0.5, 0.01, "Green: correct; orange: adjacent-stage error; red: severe cross-stage error", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_publication_figure(fig, figure_dir / "fig5_stage_error_alluvial")
    plt.close(fig)
    save_source_data(transitions, source_dir / "fig5_stage_error_alluvial.csv")


def plot_secondary_metrics(summary: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    for task in TASKS:
        subset = summary[(summary["task"] == task) & (summary["metric"] != PRIMARY_METRIC[task])].copy()
        metrics = list(subset["metric"].drop_duplicates())
        fig, axes = create_nature_figure(
            width="double", height_mm=min(210, max(70, 24 * math.ceil(len(metrics) / 3))),
            ncols=3, nrows=math.ceil(len(metrics) / 3), squeeze=False,
        )
        for ax, metric in zip(axes.ravel(), metrics):
            group = subset[subset["metric"] == metric]
            for position, strategy in enumerate(STRATEGIES):
                item = group[group["strategy"] == strategy].iloc[0]
                ax.errorbar(
                    item.estimate, position,
                    xerr=[[max(0.0, item.estimate - item.ci_low)], [max(0.0, item.ci_high - item.estimate)]],
                    fmt=STRATEGY_MARKERS[strategy], color=STRATEGY_COLORS[strategy], capsize=1.5,
                )
            ax.set_yticks(range(3), [STRATEGY_LABELS[value] for value in STRATEGIES] if ax.get_subplotspec().is_first_col() else [])
            ax.invert_yaxis()
            ax.set_title(metric.replace("_", " "))
            clean_axis(ax, grid=True)
        for ax in axes.ravel()[len(metrics):]:
            ax.set_visible(False)
        fig.tight_layout()
        save_publication_figure(fig, figure_dir / f"supp_secondary_metrics_{task}")
        plt.close(fig)
        save_source_data(subset, source_dir / f"supp_secondary_metrics_{task}.csv")


def plot_classwise_small_multiples(class_metrics: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    metrics = ["precision", "recall", "f1", "ovr_auroc", "average_precision"]
    for task in TASKS:
        subset = class_metrics[class_metrics["task"] == task]
        classes = sorted(subset["class_index"].unique())
        fig, axes = create_nature_figure(width="double", height_mm=48 * len(classes), nrows=len(classes), ncols=1, squeeze=False)
        for ax, class_index in zip(axes.ravel(), classes):
            group = subset[subset["class_index"] == class_index]
            x = np.arange(len(metrics))
            for offset, strategy in zip((-0.18, 0, 0.18), STRATEGIES):
                row = group[group["strategy"] == strategy].iloc[0]
                ax.scatter(x + offset, [row[metric] for metric in metrics], s=22, color=STRATEGY_COLORS[strategy], marker=STRATEGY_MARKERS[strategy], label=STRATEGY_LABELS[strategy])
            ax.set_xticks(x, [metric.replace("_", " ") for metric in metrics], rotation=20, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"Class {class_index}")
            clean_axis(ax)
        axes.ravel()[0].legend(ncols=3, fontsize=6)
        fig.tight_layout()
        save_publication_figure(fig, figure_dir / f"supp_classwise_{task}")
        plt.close(fig)
        save_source_data(subset, source_dir / f"supp_classwise_{task}.csv")


def plot_stage_calibration(calibration: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    fig, axes = create_nature_figure(width="double", height_mm=68, ncols=3)
    names = ["Normal", "Early", "Late"]
    for ax, class_index in zip(axes, range(3)):
        subset = calibration[calibration["class_index"] == class_index]
        for strategy in STRATEGIES:
            group = subset[subset["strategy"] == strategy]
            ax.plot(
                group["mean_predicted"], group["observed_rate"],
                color=STRATEGY_COLORS[strategy], ls=STRATEGY_LINESTYLES[strategy],
                marker=STRATEGY_MARKERS[strategy], label=STRATEGY_LABELS[strategy],
            )
        ax.plot([0, 1], [0, 1], color=COLORS["gray"], ls=":", lw=0.7)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Mean predicted probability", ylabel="Observed proportion" if class_index == 0 else "", title=names[class_index])
        clean_axis(ax)
    axes[0].legend(fontsize=6)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_stage_class_calibration")
    plt.close(fig)
    save_source_data(calibration, source_dir / "supp_stage_class_calibration.csv")


def plot_difference_rainclouds(differences: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    selected = differences[differences.apply(lambda row: row["metric"] == PRIMARY_METRIC[row["task"]], axis=1)].copy()
    fig, axes = create_nature_figure(width="double", height_mm=80, ncols=2)
    for ax, task in zip(axes, TASKS):
        subset = selected[selected["task"] == task]
        for position, ((left, right), group) in enumerate(subset.groupby(["left_strategy", "right_strategy"], sort=False)):
            values = group["difference"].to_numpy(float)
            grid, density = _density(values)
            density = density / max(density.max(), 1e-12) * 0.28
            ax.fill_between(grid, position - density, position, color=COLORS["purple"], alpha=0.25, linewidth=0)
            ax.scatter(values[::max(1, len(values) // 300)], np.full(len(values[::max(1, len(values) // 300)]), position + 0.09), s=3, color=COLORS["purple"], alpha=0.18)
            ax.plot(np.median(values), position, "o", color=COLORS["purple"], ms=4)
        ax.axvline(0, color=COLORS["gray"], ls=":", lw=0.7)
        ax.set_yticks(range(3), [f"{STRATEGY_LABELS[left]} − {STRATEGY_LABELS[right]}" for left, right in CONTRASTS])
        ax.set_xlabel(f"Bootstrap difference in {PRIMARY_METRIC[task].upper()}")
        ax.set_title("Binary classification" if task == "bin" else "Stage classification")
        ax.invert_yaxis()
        clean_axis(ax, grid=True)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_primary_difference_rainclouds")
    plt.close(fig)
    save_source_data(selected, source_dir / "supp_primary_difference_rainclouds.csv")


def plot_error_upset(overlap: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    fig, axes = create_nature_figure(width="double", height_mm=72, ncols=2)
    for ax, task in zip(axes, TASKS):
        subset = overlap[overlap["task"] == task].sort_values("count", ascending=False)
        labels = [
            " | ".join(
                f"{STRATEGY_LABELS[strategy]}:{'correct' if bit == '1' else 'error'}"
                for strategy, bit in zip(STRATEGIES, pattern)
            )
            for pattern in subset["correctness_pattern"]
        ]
        y = np.arange(len(subset))
        ax.barh(y, subset["count"], color=COLORS["blue"], alpha=0.72)
        ax.set_yticks(y, labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Hip count")
        ax.set_title("Binary" if task == "bin" else "Stage")
        clean_axis(ax)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_error_upset")
    plt.close(fig)
    save_source_data(overlap, source_dir / "supp_error_upset.csv")


def plot_laterality_dumbbells(side_table: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    selected = []
    for task in TASKS:
        metric = PRIMARY_METRIC[task]
        selected.append(side_table[(side_table["task"] == task) & (side_table["metric"] == metric)])
    data = pd.concat(selected, ignore_index=True)
    fig, axes = create_nature_figure(width="double", height_mm=66, ncols=2)
    for ax, task in zip(axes, TASKS):
        subset = data[data["task"] == task]
        for position, strategy in enumerate(STRATEGIES):
            group = subset[subset["strategy"] == strategy].set_index("side")
            left, right = group.loc["left", "estimate"], group.loc["right", "estimate"]
            ax.plot([left, right], [position, position], color=STRATEGY_COLORS[strategy], lw=1.1)
            for side, value, marker in (("left", left, "<"), ("right", right, ">")):
                item = group.loc[side]
                ax.errorbar(
                    value, position,
                    xerr=[[max(0.0, value - item.ci_low)], [max(0.0, item.ci_high - value)]],
                    fmt=marker, color=STRATEGY_COLORS[strategy], capsize=1.5,
                    label=side.title() if position == 0 else None,
                )
        ax.set_yticks(range(3), [STRATEGY_LABELS[value] for value in STRATEGIES])
        ax.invert_yaxis()
        ax.set_xlabel(PRIMARY_METRIC[task].upper())
        ax.set_title("Binary laterality" if task == "bin" else "Stage laterality")
        clean_axis(ax, grid=True)
    axes[0].legend(fontsize=6)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_laterality_dumbbells")
    plt.close(fig)
    save_source_data(data, source_dir / "supp_laterality_dumbbells.csv")


def plot_stage_error_ridges(predictions: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    data = predictions[predictions["task"] == "stage"].copy()
    data["signed_error"] = data["prediction"].astype(int) - data["label"].astype(int)
    summary = data.groupby(["strategy", "signed_error"]).size().reset_index(name="count")
    summary["proportion"] = summary.groupby("strategy")["count"].transform(lambda values: values / values.sum())
    fig, ax = create_nature_figure(width="single", height_mm=72)
    x = np.arange(-2, 3)
    for position, strategy in enumerate(STRATEGIES):
        group = summary[summary["strategy"] == strategy].set_index("signed_error")
        values = np.array([group["proportion"].get(value, 0.0) for value in x])
        baseline = position * 0.42
        ax.fill_between(x, baseline, baseline + values, color=STRATEGY_COLORS[strategy], alpha=0.28)
        ax.plot(x, baseline + values, color=STRATEGY_COLORS[strategy], lw=0.9)
        ax.text(-2.1, baseline, STRATEGY_LABELS[strategy], ha="right", va="bottom", fontsize=6)
    ax.axvline(0, color=COLORS["gray"], ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_yticks([])
    ax.set_xlabel("Predicted stage − true stage")
    clean_axis(ax)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_stage_error_ridges")
    plt.close(fig)
    save_source_data(summary, source_dir / "supp_stage_error_ridges.csv")


def plot_cross_task_agreement(detail: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    fig, axes = create_nature_figure(width="double", height_mm=122, nrows=2, ncols=3)
    for column, strategy in enumerate(STRATEGIES):
        group = detail[detail["strategy"] == strategy]
        axes[0, column].hexbin(group["stage_derived_onfh_probability"], group["prob_positive"], gridsize=16, cmap="Blues", mincnt=1, linewidths=0)
        axes[0, column].plot([0, 1], [0, 1], color=COLORS["gray"], ls=":", lw=0.7)
        axes[0, column].set(xlim=(0, 1), ylim=(0, 1), xlabel="Stage-derived P(ONFH)", ylabel="Binary P(ONFH)" if column == 0 else "", title=STRATEGY_LABELS[strategy])
        mean_prob = (group["prob_positive"] + group["stage_derived_onfh_probability"]) / 2
        difference = group["probability_difference"]
        mean_difference, sd = float(difference.mean()), float(difference.std(ddof=1))
        axes[1, column].scatter(mean_prob, difference, s=9, color=STRATEGY_COLORS[strategy], alpha=0.45, edgecolor="none")
        for value, style in ((mean_difference, "-"), (mean_difference + 1.96 * sd, "--"), (mean_difference - 1.96 * sd, "--")):
            axes[1, column].axhline(value, color=COLORS["gray"], ls=style, lw=0.7)
        axes[1, column].set(xlim=(0, 1), xlabel="Mean probability", ylabel="Binary − stage-derived probability" if column == 0 else "")
        clean_axis(axes[0, column])
        clean_axis(axes[1, column])
    add_panel_labels(axes.ravel())
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_cross_task_agreement")
    plt.close(fig)
    save_source_data(detail, source_dir / "supp_cross_task_agreement.csv")


def high_confidence_failures(predictions: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    data = predictions.copy()
    data["confidence"] = np.where(
        data["task"] == "bin",
        np.where(data["prediction"] == 1, data.get("prob_positive", np.nan), 1 - data.get("prob_positive", np.nan)),
        data[[f"prob_class_{index}" for index in range(3)]].max(axis=1, skipna=True),
    )
    failures = data[data["prediction"].astype(int) != data["label"].astype(int)].copy()
    return failures.sort_values(["task", "strategy", "confidence"], ascending=[True, True, False]).groupby(["task", "strategy"], as_index=False).head(limit)


def plot_high_confidence_failure_matrix(failures: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    summary = failures.groupby(["task", "strategy"]).agg(
        error_count=("confidence", "size"),
        mean_confidence=("confidence", "mean"),
        max_confidence=("confidence", "max"),
    ).reset_index()
    fig, axes = create_nature_figure(width="double", height_mm=58, ncols=2)
    for ax, task in zip(axes, TASKS):
        group = summary[summary["task"] == task].set_index("strategy").reindex(STRATEGIES)
        matrix = group[["error_count", "mean_confidence", "max_confidence"]].to_numpy(float)
        normalized = matrix / np.clip(np.nanmax(matrix, axis=0, keepdims=True), 1e-12, None)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row in range(3):
            for column in range(3):
                ax.text(column, row, f"{matrix[row, column]:.2f}" if column else f"{int(matrix[row, column])}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(3), ["Errors", "Mean confidence", "Max confidence"], rotation=25, ha="right")
        ax.set_yticks(range(3), [STRATEGY_LABELS[value] for value in STRATEGIES])
        ax.set_title("Binary" if task == "bin" else "Stage")
        ax.tick_params(length=0)
    add_panel_labels(axes)
    fig.tight_layout()
    save_publication_figure(fig, figure_dir / "supp_high_confidence_failure_matrix")
    plt.close(fig)
    save_source_data(summary, source_dir / "supp_high_confidence_failure_matrix.csv")


def _write_analysis_note(output_dir: Path, n_bootstrap: int, n_permutations: int, seed: int) -> None:
    note = f"""# Combined external-protocol statistical analysis

## Material Passport

- Verification Status: ANALYZED
- Analysis unit: patient-clustered; bilateral hips stay together
- Bootstrap replicates: {n_bootstrap}
- Paired permutations: {n_permutations}
- Random seed: {seed}
- Binary operating threshold: 0.5 (locked; not tuned on UIH)

## Interpretation boundary

zero_shot is the independent cross-device/protocol evaluation. siemens_finetune
and uih_ssl_only are target-domain out-of-fold adaptation analyses and must not be
described as independent zero-shot external validation. Bootstrap densities represent
uncertainty in estimates, not the distribution of individual patients.

## Multiplicity and uncertainty

All confidence intervals use patient-cluster percentile bootstrap. The three primary
contrasts within each task use Holm adjustment. Secondary endpoints are exploratory
and use Benjamini-Hochberg FDR within task. Exact effect estimates and confidence
intervals should be interpreted before adjusted P values.

## Statistical fallacy scan (11/11 checked)

- Simpson/ecological: aggregate, laterality and class-specific results are separated;
  no individual inference is made from group-only data.
- Berkson/collider: this evaluation does not condition on model predictions or
  post-outcome controls; cohort selection remains a study-design limitation.
- Base-rate neglect: prevalence-dependent PPV, NPV and AUPRC accompany sensitivity,
  specificity and AUROC.
- Regression to the mean: no cohort was selected by extreme model performance.
- Survivorship: completeness audit requires every strategy to cover the same patients.
- Look-elsewhere/forking paths: primary metrics and contrasts are pre-specified;
  secondary analyses are labelled exploratory and multiplicity-adjusted.
- Correlation/causation and reverse causality: results describe predictive performance
  and consistency only; no causal claims are supported.
"""
    (output_dir / "analysis_methods_and_interpretation.md").write_text(note, encoding="utf-8")


def run_combined_analysis(
    analysis_root: str | Path,
    output_dir: str | Path | None = None,
    n_bootstrap: int = 5000,
    n_permutations: int = 10000,
    seed: int = 2026,
    show_progress: bool = True,
) -> dict:
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20.")
    if n_permutations < 20:
        raise ValueError("n_permutations must be at least 20.")
    analysis_root = Path(analysis_root)
    output_dir = Path(output_dir) if output_dir else analysis_root / "combined_statistical_analysis"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    for directory in (output_dir, tables_dir, figures_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    overall_progress = tqdm(
        total=8,
        desc="联合分析总进度",
        unit="阶段",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    predictions, audit = load_analysis_predictions(analysis_root)
    predictions.to_csv(tables_dir / "combined_predictions_long.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(tables_dir / "data_integrity_audit.csv", index=False, encoding="utf-8-sig")
    overall_progress.set_postfix_str("数据读取与完整性审计完成")
    overall_progress.update(1)

    summary, bootstrap_distribution = bootstrap_metrics(
        predictions, n_bootstrap, seed, show_progress=show_progress
    )
    overall_progress.set_postfix_str("总体指标与聚类 bootstrap 完成")
    overall_progress.update(1)
    comparisons, difference_distribution = paired_strategy_comparisons(
        predictions,
        bootstrap_distribution,
        n_permutations,
        seed,
        show_progress=show_progress,
    )
    overall_progress.set_postfix_str("三策略配对比较与置换检验完成")
    overall_progress.update(1)
    classes = classwise_metrics(predictions)
    sides = side_metrics(
        predictions,
        n_bootstrap=n_bootstrap,
        seed=seed,
        show_progress=show_progress,
    )
    stage_calibration = stage_calibration_table(predictions)
    overlap = error_overlap(predictions)
    transitions = stage_error_transitions(predictions)
    consistency, consistency_detail = cross_task_consistency(predictions)
    failures = high_confidence_failures(predictions)
    overall_progress.set_postfix_str("类别、侧别、校准及错误分析完成")
    overall_progress.update(1)

    outputs = {
        "overall_metrics": summary,
        "paired_strategy_comparisons": comparisons,
        "bootstrap_distribution": bootstrap_distribution,
        "paired_difference_distribution": difference_distribution,
        "classwise_metrics": classes,
        "side_metrics": sides,
        "stage_class_calibration": stage_calibration,
        "error_overlap": overlap,
        "stage_error_transitions": transitions,
        "cross_task_consistency": consistency,
        "cross_task_consistency_detail": consistency_detail,
        "high_confidence_failures": failures,
    }
    for name, frame in outputs.items():
        if frame.columns.duplicated().any():
            raise RuntimeError(f"Duplicate output columns detected in {name}.")
        frame.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    overall_progress.set_postfix_str("统计表保存完成")
    overall_progress.update(1)

    figure_jobs = [
        ("主指标半眼图", lambda: plot_primary_half_eye(summary, bootstrap_distribution, figures_dir, source_dir)),
        ("Gardner–Altman估计图", lambda: plot_gardner_altman(summary, comparisons, figures_dir, source_dir)),
        ("二分类诊断组合图", lambda: plot_binary_diagnostics(predictions, figures_dir, source_dir)),
        ("分期气泡混淆矩阵", lambda: plot_stage_bubble_confusions(predictions, figures_dir, source_dir)),
        ("分期误差迁移流图", lambda: plot_stage_alluvial(transitions, figures_dir, source_dir)),
        ("次要指标图", lambda: plot_secondary_metrics(summary, figures_dir, source_dir)),
        ("类别级性能图", lambda: plot_classwise_small_multiples(classes, figures_dir, source_dir)),
        ("分期校准图", lambda: plot_stage_calibration(stage_calibration, figures_dir, source_dir)),
        ("配对差值雨云图", lambda: plot_difference_rainclouds(difference_distribution, figures_dir, source_dir)),
        ("错误组合UpSet图", lambda: plot_error_upset(overlap, figures_dir, source_dir)),
        ("左右侧哑铃图", lambda: plot_laterality_dumbbells(sides, figures_dir, source_dir)),
        ("分期误差脊线图", lambda: plot_stage_error_ridges(predictions, figures_dir, source_dir)),
        ("跨任务一致性图", lambda: plot_cross_task_agreement(consistency_detail, figures_dir, source_dir)),
        ("高置信度失败矩阵", lambda: plot_high_confidence_failure_matrix(failures, figures_dir, source_dir)),
    ]
    for figure_name, plot_job in tqdm(
        figure_jobs,
        desc="生成 Nature 风格图形",
        unit="幅",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    ):
        if show_progress:
            overall_progress.set_postfix_str(f"正在绘制：{figure_name}")
        plot_job()
    overall_progress.set_postfix_str("图形及source-data生成完成")
    overall_progress.update(1)

    figure_audit_rows = []
    for path in sorted(figures_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".svg", ".png"}:
            continue
        issues = audit_figure(path)
        figure_audit_rows.append(
            {"figure": str(path), "format": path.suffix.lower().lstrip("."), "passed": not issues, "issues": " | ".join(issues)}
        )
    figure_audit = pd.DataFrame(figure_audit_rows)
    figure_audit.to_csv(tables_dir / "figure_audit.csv", index=False, encoding="utf-8-sig")
    _write_analysis_note(output_dir, n_bootstrap, n_permutations, seed)
    overall_progress.set_postfix_str("图形审计与分析说明完成")
    overall_progress.update(1)

    manifest = {
        "analysis": "two_tasks_three_strategies_patient_clustered",
        "analysis_root": str(analysis_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "tasks": list(TASKS),
        "strategies": list(STRATEGIES),
        "primary_metrics": PRIMARY_METRIC,
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "binary_threshold": 0.5,
        "n_patients": int(predictions["patient_id"].nunique()),
        "n_hips_per_strategy_task": int(predictions.groupby(["task", "strategy"]).size().min()),
        "input_files": audit[["task", "strategy", "evaluation_design", "source_file", "sha256"]].to_dict("records"),
        "interpretation_boundary": (
            "zero_shot is external cross-device/protocol evaluation; "
            "siemens_finetune and uih_ssl_only are target-domain OOF adaptation"
        ),
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    overall_progress.set_postfix_str("全部完成")
    overall_progress.update(1)
    overall_progress.close()
    return {
        "output_dir": str(output_dir),
        "n_patients": manifest["n_patients"],
        "n_hips_per_strategy_task": manifest["n_hips_per_strategy_task"],
        "figure_audit_passed": bool(figure_audit["passed"].all()) if not figure_audit.empty else False,
        "primary_results": summary[
            summary.apply(lambda row: row["metric"] == PRIMARY_METRIC[row["task"]], axis=1)
        ].to_dict("records"),
    }
