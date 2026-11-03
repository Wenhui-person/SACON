
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
except (AttributeError, TypeError, ValueError, OSError):
    pass
print("[启动] SACON 消融流程正在加载依赖，请稍候……", flush=True)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from configs.config import TrainConfig, get_ablation_configs
from data.dataset import ONFHDataset
from inference import ordinal_probs_from_logits
from models.sacon import SACON
from train import build_cv_splits, run_experiment, save_fold_split_csv
from utils.publication_style import (
    COLORS,
    STAGE_COLORS,
    audit_figure,
    clean_axis,
    create_nature_figure,
    save_publication_figure,
)

DEFAULT_SPEC = "configs/stage_ablation_no_conc.json"
PROJECT_ROOT = Path(__file__).resolve().parent

IDE_RUN_CONFIG = {
    "spec_path": DEFAULT_SPEC,
    "steps": ["visualize-mri"],
    "variants": ["all"],
    "folds": [0, 1, 2, 3, 4],
    "device": "cuda",
    "n_bootstrap": 5000,
    "resume": True,
    "inference_batch_size": 2,
    "enable_gradient_attribution": True,
    "run_contralateral_interventions": True,
    "intervention_visualization_cases": 5,
}

PREDICTION_COLUMNS = [
    "experiment_id", "variant", "fold", "checkpoint", "patient_id",
    "source_index", "side", "label", "prediction", "prob_class_0",
    "prob_class_1", "prob_class_2", "confidence", "abs_error",
]


class ProgressReporter:

    def __init__(self, label: str, total: int, updates: int = 10):
        self.label = label
        self.total = max(int(total), 1)
        self.interval = max(1, math.ceil(self.total / max(updates, 1)))
        self.last_reported = 0
        print(f"[进度] {self.label}：开始，共 {total} 项", flush=True)

    def update(self, completed: int, detail: str = "") -> None:
        completed = min(int(completed), self.total)
        if completed != self.total and completed - self.last_reported < self.interval:
            return
        self.last_reported = completed
        percent = 100.0 * completed / self.total
        suffix = f"｜{detail}" if detail else ""
        print(f"[进度] {self.label}：{completed}/{self.total}（{percent:.0f}%）{suffix}", flush=True)


def configure_live_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(
                encoding="utf-8", errors="replace",
                line_buffering=True, write_through=True,
            )
        except (TypeError, ValueError, OSError):
            pass


def load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    loss = spec["loss"]
    if loss["alpha_concentration"] != 0 or loss["alpha_local_concentration"] != 0:
        raise ValueError("本研究已约定不使用集中性损失；两个 concentration 权重必须均为 0。")
    split = spec["split"]
    if split["mode"] != "repeated_random_holdout" or split["n_repeats"] != 5:
        raise ValueError("数据划分必须保持 train.py 的 5 次重复随机留出规则。")
    unknown = set(spec["variants"]) - set(get_ablation_configs())
    if unknown:
        raise ValueError(f"配置包含未知变体: {sorted(unknown)}")
    return spec


def experiment_dir(spec: dict) -> Path:
    return Path(spec["output_root"]) / spec["experiment_id"]


def ensure_layout(spec: dict) -> Path:
    root = experiment_dir(spec)
    for relative in (
        "splits", "checkpoints", "logs", "predictions", "metrics", "mechanisms",
        "figures/main", "figures/supplementary", "source_data", "diagnostics",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_splits(spec: dict, data: dict):
    return build_cv_splits(
        data=data,
        task="stage",
        group_by_base_id=False,
        repeated_holdout_7_3=False,
        real_only_test=True,
        real_test_ratio=float(spec["split"]["real_test_ratio"]),
    )


def effective_stage_parameters(model: SACON) -> int:
    excluded = ["left_bin_head", "right_bin_head"]
    config = model.config
    if not config.use_contralateral or not config.use_explicit_difference:
        excluded.extend(["contralateral", "left_fusion_gate", "right_fusion_gate"])
    elif not config.use_structure_signal_decoupling:
        excluded.extend(["left_fusion_gate", "right_fusion_gate"])
    elif not config.use_local_diff_gate:
        excluded.extend(["left_fusion_gate.gate", "right_fusion_gate.gate"])
    return sum(parameter.numel() for name, parameter in model.named_parameters()
               if not any(name.startswith(prefix) for prefix in excluded))


def prepare(spec: dict, spec_path: str) -> None:
    root = ensure_layout(spec)
    data = load_manifest(spec["data_root"])
    splits = canonical_splits(spec, data)
    progress = ProgressReporter("生成固定数据划分", len(splits))
    for fold, (train_index, test_index) in enumerate(splits):
        fold_dir = root / "splits" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_fold_split_csv(data, train_index, test_index, fold, str(fold_dir))
        progress.update(fold + 1, f"fold {fold}")

    base_channels = int(spec["training"]["base_channels"])
    fingerprint = hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
    run_manifest = {
        "experiment_id": spec["experiment_id"],
        "spec_sha256": fingerprint,
        "task": "three-class ordinal staging",
        "validation_design": "five-repetition random holdout validation",
        "model_selection": "validation accuracy; lower validation loss breaks ties",
        "seed": spec["seed"],
        "concentration_losses_enabled": False,
        "base_channels": base_channels,
        "patient_count": len(data["patient_id"]),
        "privacy": "输出仅含去标识 patient_id；固定病例在图中显示为 Case 1–5。",
    }
    with open(root / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)
    print(f"[prepare] 已创建实验目录: {root}")


def dry_run(spec: dict) -> None:
    root = ensure_layout(spec)
    data = load_manifest(spec["data_root"])
    splits = canonical_splits(spec, data)
    base_channels = int(spec["training"]["base_channels"])
    rows = []
    progress = ProgressReporter("检查消融模型配置", len(spec["variants"]))
    for index, name in enumerate(spec["variants"]):
        model = SACON(get_ablation_configs()[name], base_channels=base_channels,
                      num_stages=3, ordinal=True, task="stage")
        rows.append({
            "variant": name,
            "base_channels": base_channels,
            "effective_stage_parameters": effective_stage_parameters(model),
            **asdict(get_ablation_configs()[name]),
        })
        progress.update(index + 1, name)
    pd.DataFrame(rows).to_csv(root / "diagnostics" / "model_parameters.csv", index=False)
    split_rows = [{"fold": i, "train_n": len(a), "holdout_n": len(b)} for i, (a, b) in enumerate(splits)]
    pd.DataFrame(split_rows).to_csv(root / "diagnostics" / "split_summary.csv", index=False)
    print(pd.DataFrame(rows)[["variant", "base_channels", "effective_stage_parameters"]].to_string(index=False))


def train_variants(spec: dict, variants: list[str], folds: tuple[int, ...], resume: bool = False) -> None:
    root = ensure_layout(spec)
    base_channels = int(spec["training"]["base_channels"])
    tr = spec["training"]
    loss = spec["loss"]
    ssl_path = spec.get("ssl_checkpoint") or None
    progress = ProgressReporter("训练消融变体", len(variants))
    for variant_index, variant in enumerate(variants):
        selected_folds = folds
        if resume:
            incomplete_folds = []
            for fold in folds:
                fold_dir = root / "checkpoints" / variant / f"fold_{fold}"
                histories = list(fold_dir.glob(f"stage_{variant}*_history.json"))
                try:
                    checkpoint_path(root, variant, fold)
                    has_best = True
                except (FileNotFoundError, RuntimeError):
                    has_best = False
                if not (has_best and len(histories) == 1):
                    incomplete_folds.append(fold)
            selected_folds = tuple(incomplete_folds)
            if not selected_folds:
                print(f"[resume] variant={variant}: all requested folds are complete; skipping")
                progress.update(variant_index + 1, f"{variant} 已完成，跳过")
                continue
        cfg = TrainConfig(
            data_root=spec["data_root"], base_channels=base_channels,
            num_epochs=int(tr["epochs"]), batch_size=int(tr["batch_size"]),
            lr=float(tr["learning_rate"]), num_workers=int(tr["num_workers"]),
            pin_memory=bool(tr["pin_memory"]),
            persistent_workers=bool(tr["persistent_workers"]),
            prefetch_factor=int(tr["prefetch_factor"]),
            volume_size=tuple(tr["volume_size"]), patience=int(tr["patience"]),
            task="stage", experiment_root=str(root), selected_folds=selected_folds,
            save_dir=str(root / "logs" / variant),
            random_seed=int(spec["seed"]), ssl_t1_ckpt=ssl_path, ssl_t2_ckpt=ssl_path,
            freeze_encoder_epochs=int(tr["freeze_encoder_epochs"]),
            alpha_stage=float(loss["alpha_stage"]),
            alpha_reg=float(loss["alpha_registration"]),
            alpha_decouple=float(loss["alpha_decoupling"]),
            alpha_conc=0.0, alpha_local_conc=0.0,
            real_only_test=True, real_test_ratio=float(spec["split"]["real_test_ratio"]),
            feature_cloud_interval=int(tr["feature_cloud_interval"]),
            feature_cloud_3d=bool(tr["feature_cloud_3d"]),
            save_best_feature_cloud=bool(tr["save_best_feature_cloud"]),
            grad_flow_interval=int(tr["grad_flow_interval"]),
            grad_flow_samples=int(tr["grad_flow_samples"]),
            tensorboard_image_interval=int(tr["tensorboard_image_interval"]),
            progress_update_interval=int(tr["progress_update_interval"]),
            enable_cam_hooks=bool(tr["enable_cam_hooks"]),
            plot_interval=int(tr["plot_interval"]),
            plot_on_best=bool(tr["plot_on_best"]),
        )
        cfg.resume_from_checkpoint = resume
        print(f"[train] variant={variant}, folds={selected_folds}, base_channels={base_channels}, resume={resume}")
        run_experiment(variant, cfg)
        progress.update(variant_index + 1, variant)


def checkpoint_path(root: Path, variant: str, fold: int) -> Path:
    fold_dir = root / "checkpoints" / variant / f"fold_{fold}"
    exact = fold_dir / f"stage_{variant}_best.pth"
    if exact.is_file():
        return exact

    tagged = sorted(fold_dir.glob(f"stage_{variant}_*_best.pth")) if fold_dir.is_dir() else []
    if len(tagged) == 1:
        return tagged[0]
    if len(tagged) > 1:
        names = ", ".join(path.name for path in tagged)
        raise RuntimeError(
            f"Ambiguous checkpoints for variant={variant}, fold={fold}: {names}. "
            "请删除重复文件或在配置中统一 run_tag。"
        )
    raise FileNotFoundError(
        f"Missing checkpoint for variant={variant}, fold={fold}. "
        f"Checked: {exact} and pattern stage_{variant}_*_best.pth"
    )


def resolve_checkpoint_matrix(root: Path, variants: Iterable[str],
                              folds: Iterable[int]) -> dict[tuple[str, int], Path]:
    variants = list(variants)
    folds = list(folds)
    resolved = {}
    errors = []
    progress = ProgressReporter("检查最佳 checkpoint", len(variants) * len(folds))
    completed = 0
    for variant in variants:
        for fold in folds:
            try:
                resolved[(variant, int(fold))] = checkpoint_path(root, variant, int(fold))
            except (FileNotFoundError, RuntimeError) as exc:
                errors.append(str(exc))
            completed += 1
            progress.update(completed, f"{variant} / fold {fold}")
    if errors:
        details = "\n  - ".join(errors)
        raise FileNotFoundError(
            "Checkpoint preflight failed; inference has not started.\n  - " + details
        )
    return resolved


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint(path: Path, device: torch.device) -> tuple[SACON, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    config_values = checkpoint["ablation_cfg"]
    config = get_ablation_configs().get("full").__class__(**config_values)
    train_cfg = checkpoint.get("train_cfg", {})
    model = SACON(config, base_channels=int(train_cfg.get("base_channels", 32)),
                  num_stages=3, ordinal=True, task="stage")
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "train_cfg": train_cfg,
        "ablation_cfg": config_values,
        "val_metrics": checkpoint.get("val_metrics", {}),
    }
    del checkpoint
    model.to(device)
    model.eval()
    return model, metadata


def predict_loader(model: SACON, loader: DataLoader, device: torch.device,
                   experiment_id: str, variant: str, fold: int, checkpoint: str,
                   enable_gradient_attribution: bool = False):
    def normalized_cross_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float().reshape(-1) - a.float().mean()
        b = b.float().reshape(-1) - b.float().mean()
        return float((a @ b / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-8)).detach())

    def spatial_entropy(value: torch.Tensor) -> float:
        weights = value.detach().float().abs().reshape(-1)
        weights = weights / (weights.sum() + 1e-8)
        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        return float(entropy / math.log(max(weights.numel(), 2)))

    def negative_jacobian_fraction(flow: torch.Tensor) -> float:
        array = flow.detach().float().cpu().numpy()
        derivatives = [[np.gradient(array[i], axis=j) for j in range(3)] for i in range(3)]
        jacobian = np.empty(array.shape[1:] + (3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                jacobian[..., i, j] = derivatives[i][j] + (1.0 if i == j else 0.0)
        return float(np.mean(np.linalg.det(jacobian) <= 0))

    rows, mechanism_rows = [], []
    progress = ProgressReporter(f"内部推理 {variant} / fold {fold}", len(loader))
    for batch_index, batch in enumerate(loader):
        tensors = [batch[key].to(device) for key in ("left_t1", "left_t2", "right_t1", "right_t2")]
        contributions = {}
        if enable_gradient_attribution:
            with torch.enable_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(*tensors)
                feature_items = []
                for side in ("left", "right"):
                    for feature_name in ("local", "diff"):
                        feature = outputs.get(f"z_{side}_{feature_name}")
                        if feature is not None:
                            feature_items.append((side, feature_name, feature))
                if feature_items:
                    target = outputs["left_stage"].sum() + outputs["right_stage"].sum()
                    gradients = torch.autograd.grad(
                        target, [item[2] for item in feature_items], allow_unused=True)
                    for (side, feature_name, feature), gradient in zip(feature_items, gradients):
                        if gradient is not None:
                            values = (gradient * feature).abs().flatten(1).mean(1)
                            contributions.setdefault(side, {})[feature_name] = (
                                values.detach().cpu().numpy())
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(*tensors)
        side_probs = {
            "left": ordinal_probs_from_logits(outputs["left_stage"], 3).detach().cpu().numpy(),
            "right": ordinal_probs_from_logits(outputs["right_stage"], 3).detach().cpu().numpy(),
        }
        for side, probs in side_probs.items():
            labels = batch[f"{side}_stage"].numpy()
            for i, probability in enumerate(probs):
                prediction = int(np.argmax(probability))
                rows.append({
                    "experiment_id": experiment_id, "variant": variant, "fold": fold,
                    "checkpoint": checkpoint, "patient_id": str(batch["patient_id"][i]),
                    "source_index": int(batch["source_index"][i]), "side": side,
                    "label": int(labels[i]), "prediction": prediction,
                    "prob_class_0": float(probability[0]), "prob_class_1": float(probability[1]),
                    "prob_class_2": float(probability[2]), "confidence": float(np.max(probability)),
                    "abs_error": abs(prediction - int(labels[i])),
                })

        batch_size = len(batch["patient_id"])
        for i in range(batch_size):
            item = {"variant": variant, "fold": fold, "patient_id": str(batch["patient_id"][i])}
            left_label = int(batch["left_stage"][i])
            right_label = int(batch["right_stage"][i])
            if left_label == right_label == 0:
                phenotype = "bilateral_normal"
            elif (left_label == 0) != (right_label == 0):
                phenotype = "unilateral"
            elif left_label == right_label:
                phenotype = "concordant_bilateral"
            else:
                phenotype = "discordant_bilateral"
            item["phenotype"] = phenotype
            for key in ("modal_gate_left", "modal_gate_right", "left_diff_gate", "right_diff_gate"):
                value = outputs.get(key)
                item[f"{key}_mean"] = float(value[i].mean()) if value is not None else np.nan
                item[f"{key}_std"] = float(value[i].std()) if value is not None else np.nan
                item[f"{key}_spatial_entropy"] = spatial_entropy(value[i]) if value is not None else np.nan
            for side in ("left", "right"):
                structure, signal = outputs.get(f"f_{side}_str"), outputs.get(f"f_{side}_sig")
                if structure is not None and signal is not None:
                    s = structure[i].reshape(-1)
                    g = signal[i].reshape(-1)
                    item[f"{side}_structure_l2"] = float(torch.linalg.vector_norm(s))
                    item[f"{side}_signal_l2"] = float(torch.linalg.vector_norm(g))
                    item[f"{side}_abs_cosine"] = float(torch.abs(torch.nn.functional.cosine_similarity(s, g, dim=0)))
                if side in contributions:
                    local_contribution = contributions[side].get("local")
                    diff_contribution = contributions[side].get("diff")
                    item[f"{side}_local_grad_activation"] = (
                        float(local_contribution[i]) if local_contribution is not None else np.nan)
                    item[f"{side}_diff_grad_activation"] = (
                        float(diff_contribution[i]) if diff_contribution is not None else np.nan)
                else:
                    item[f"{side}_local_grad_activation"] = np.nan
                    item[f"{side}_diff_grad_activation"] = np.nan
            flow = outputs.get("disp_fields")
            item["displacement_mean"] = float(flow[i].abs().mean()) if flow is not None else np.nan
            item["displacement_max"] = float(flow[i].abs().max()) if flow is not None else np.nan
            item["negative_jacobian_fraction"] = negative_jacobian_fraction(flow[i]) if flow is not None else np.nan
            before_right_t1 = torch.flip(tensors[2][i], dims=[-1])
            before_right_t2 = torch.flip(tensors[3][i], dims=[-1])
            ncc_before = 0.5 * (
                normalized_cross_correlation(tensors[0][i], before_right_t1)
                + normalized_cross_correlation(tensors[1][i], before_right_t2)
            )
            ncc_after = 0.5 * (
                normalized_cross_correlation(outputs["xl_t1_aligned"][i], outputs["xr_t1_aligned"][i])
                + normalized_cross_correlation(outputs["xl_t2_aligned"][i], outputs["xr_t2_aligned"][i])
            )
            item["ncc_before"] = ncc_before
            item["ncc_after"] = ncc_after
            item["ncc_improvement"] = ncc_after - ncc_before
            mechanism_rows.append(item)
        del outputs, tensors
        progress.update(batch_index + 1)
    return rows, mechanism_rows


def infer_internal(spec: dict, variants: list[str], folds: tuple[int, ...], device_name: str,
                   inference_batch_size: int = 1,
                   enable_gradient_attribution: bool = False,
                   run_interventions: bool = True) -> None:
    root = ensure_layout(spec)
    data = load_manifest(spec["data_root"])
    splits = canonical_splits(spec, data)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    checkpoint_map = resolve_checkpoint_matrix(root, variants, folds)
    pd.DataFrame([
        {"variant": variant, "fold": fold, "checkpoint": str(path)}
        for (variant, fold), path in checkpoint_map.items()
    ]).to_csv(root / "diagnostics" / "resolved_checkpoints.csv", index=False)
    all_rows, all_mechanisms = [], []
    all_interventions = []
    overall = ProgressReporter("内部推理总进度", len(variants) * len(folds))
    completed_models = 0
    for variant in variants:
        for fold in folds:
            path = checkpoint_map[(variant, fold)]
            model, _ = load_checkpoint(path, device)
            dataset = ONFHDataset(spec["data_root"], splits[fold][1],
                                  volume_size=tuple(spec["training"]["volume_size"]), augment=False)
            loader = DataLoader(
                dataset, batch_size=inference_batch_size, shuffle=False,
                num_workers=int(spec["training"]["num_workers"]),
                pin_memory=bool(spec["training"].get("pin_memory", False)),
            )
            rows, mechanisms = predict_loader(model, loader, device, spec["experiment_id"],
                                               variant, fold, str(path),
                                               enable_gradient_attribution)
            all_rows.extend(rows)
            all_mechanisms.extend(mechanisms)
            if variant == "full" and run_interventions:
                all_interventions.extend(
                    contralateral_interventions(
                        model, dataset, device, fold, int(spec["seed"]),
                    )
                )
            del model, loader, dataset
            clear_cuda_cache()
            completed_models += 1
            overall.update(completed_models, f"{variant} / fold {fold}")
    pd.DataFrame(all_rows, columns=PREDICTION_COLUMNS).to_csv(
        root / "predictions" / "internal_holdout_predictions.csv", index=False)
    pd.DataFrame(all_mechanisms).to_csv(root / "mechanisms" / "internal_mechanism_summary.csv", index=False)
    pd.DataFrame(all_interventions).to_csv(
        root / "mechanisms" / "contralateral_interventions_full.csv", index=False)
    print(f"[infer-internal] 导出 {len(all_rows)} 条髋关节预测。")


def _robust_image_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = [np.asarray(image)[np.isfinite(image)] for image in images if np.asarray(image).size]
    values = np.concatenate([value.ravel() for value in finite if value.size])
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(values, [0.01, 0.99])
    return float(low), float(max(high, low + 1e-8))


def _save_intervention_mri_panel(target_images: tuple[np.ndarray, np.ndarray],
                                 condition_images: dict[str, tuple[np.ndarray, np.ndarray]],
                                 condition_probabilities: dict[str, float],
                                 output_stem: Path,
                                 target_side: str,
                                 reference_stage: int,
                                 slice_index: int) -> None:
    import matplotlib.pyplot as plt

    order = [
        "target", "correct", "random_wrong_patient",
        "stage_matched_wrong_patient", "zero", "self_copy",
    ]
    titles = {
        "target": f"Target ({target_side.upper()})",
        "correct": "Correct contralateral",
        "random_wrong_patient": "Random donor",
        "stage_matched_wrong_patient": "Stage-matched donor",
        "zero": "Zero input",
        "self_copy": "Self-copy",
    }
    images = {"target": target_images, **condition_images}
    target_marker = target_side[0].upper()
    opposite_marker = ("R" if target_side == "left" else "L")
    nonzero_names = [name for name in order if name != "zero"]
    t1_vmin, t1_vmax = _robust_image_limits([images[name][0] for name in nonzero_names])
    t2_vmin, t2_vmax = _robust_image_limits([images[name][1] for name in nonzero_names])
    fig, axes = create_nature_figure(
        "double", 62, nrows=2, ncols=3,
        gridspec_kw={"wspace": 0.04, "hspace": 0.25},
    )
    for ax, name in zip(np.asarray(axes).ravel(), order):
        t1_image, t2_image = images[name]
        t1_display = np.clip((t1_image - t1_vmin) / (t1_vmax - t1_vmin), 0.0, 1.0)
        t2_display = np.clip((t2_image - t2_vmin) / (t2_vmax - t2_vmin), 0.0, 1.0)
        paired = np.concatenate([t1_display, t2_display], axis=1)
        ax.imshow(paired, cmap="gray", vmin=0, vmax=1)
        ax.axvline(t1_image.shape[1] - 0.5, color="white", linewidth=0.4, alpha=0.8)
        side_marker = target_marker if name == "target" else opposite_marker
        ax.text(
            0.025, 0.96, side_marker, transform=ax.transAxes,
            color="white", fontsize=6.5, fontweight="bold", ha="left", va="top",
        )
        ax.text(0.02, 0.04, "T1", transform=ax.transAxes, color="white",
                fontsize=5.5, va="bottom")
        ax.text(0.52, 0.04, "T2", transform=ax.transAxes, color="white",
                fontsize=5.5, va="bottom")
        probability = condition_probabilities.get(name)
        subtitle = titles[name]
        if probability is not None:
            subtitle += f"\n$P$(stage {reference_stage})={probability:.2f}"
        ax.set_title(subtitle, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Contralateral intervention, locked slice z={slice_index}", fontsize=8, y=1.01)
    save_publication_figure(fig, output_stem)
    plt.close(fig)


@torch.no_grad()
def contralateral_interventions(model: SACON, dataset: ONFHDataset, device: torch.device,
                                fold: int, seed: int,
                                figure_dir: Path | None = None,
                                max_visual_cases: int = 0,
                                max_cases: int | None = None) -> list[dict]:
    rng = np.random.default_rng(seed + fold)
    sample_count = len(dataset)
    if sample_count < 2:
        raise ValueError("对侧干预至少需要 2 个患者样本，才能构造错误患者对照。")
    source_indices = [int(index) for index in dataset.ids]
    labels = {
        side: np.asarray([
            int(dataset.samples[f"label_{side}_ELS"][source_index])
            for source_index in source_indices
        ])
        for side in ("left", "right")
    }
    processing_count = sample_count if max_cases is None else min(max(int(max_cases), 0), sample_count)
    records = []
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(f"对侧干预 Full / fold {fold}", processing_count)
    for index in range(processing_count):
        sample = dataset[index]
        for target_side in ("left", "right"):
            opposite = "right" if target_side == "left" else "left"
            candidates = np.delete(np.arange(sample_count), index)
            random_index = int(rng.choice(candidates))
            matched = candidates[labels[opposite][candidates] == labels[opposite][index]]
            matched_index = int(rng.choice(matched)) if len(matched) else random_index
            conditions = {
                "correct": None,
                "random_wrong_patient": random_index,
                "stage_matched_wrong_patient": matched_index,
                "zero": "zero",
                "self_copy": "self",
            }
            correct_probability = None
            condition_images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            condition_probabilities: dict[str, float] = {}
            target_volume = sample[f"{target_side}_t1"][0].detach().cpu().numpy()
            slice_index = int(target_volume.shape[0] // 2)
            for condition, donor in conditions.items():
                inputs = {
                    key: sample[key].unsqueeze(0).to(device)
                    for key in ("left_t1", "left_t2", "right_t1", "right_t2")
                }
                if isinstance(donor, int):
                    donor_sample = dataset[donor]
                    inputs[f"{opposite}_t1"] = donor_sample[f"{opposite}_t1"].unsqueeze(0).to(device)
                    inputs[f"{opposite}_t2"] = donor_sample[f"{opposite}_t2"].unsqueeze(0).to(device)
                elif donor == "zero":
                    inputs[f"{opposite}_t1"] = torch.zeros_like(inputs[f"{opposite}_t1"])
                    inputs[f"{opposite}_t2"] = torch.zeros_like(inputs[f"{opposite}_t2"])
                elif donor == "self":
                    inputs[f"{opposite}_t1"] = inputs[f"{target_side}_t1"].clone()
                    inputs[f"{opposite}_t2"] = inputs[f"{target_side}_t2"].clone()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output = model(inputs["left_t1"], inputs["left_t2"],
                                   inputs["right_t1"], inputs["right_t2"])
                probability = ordinal_probs_from_logits(output[f"{target_side}_stage"], 3)[0].cpu().numpy()
                reference = int(sample[f"{target_side}_stage"])
                reference_probability = float(probability[reference])
                condition_probabilities[condition] = reference_probability
                if figure_dir is not None and index < max_visual_cases:
                    condition_images[condition] = (
                        inputs[f"{opposite}_t1"][0, 0, slice_index].float().cpu().numpy(),
                        inputs[f"{opposite}_t2"][0, 0, slice_index].float().cpu().numpy(),
                    )
                if condition == "correct":
                    correct_probability = reference_probability
                records.append({
                    "fold": fold, "patient_id": str(sample["patient_id"]), "target_side": target_side,
                    "condition": condition, "label": reference, "prediction": int(np.argmax(probability)),
                    "reference_class_probability": reference_probability,
                    "delta_reference_probability_vs_correct": (
                        reference_probability - correct_probability if correct_probability is not None else 0.0),
                    "visualized": bool(figure_dir is not None and index < max_visual_cases),
                    "locked_slice": slice_index,
                    "donor_patient_id": (
                        str(dataset.samples["patient_id"][source_indices[donor]])
                        if isinstance(donor, int) else str(donor or "correct")),
                })
                del output, inputs
            if figure_dir is not None and index < max_visual_cases:
                output_stem = figure_dir / f"case_{index + 1:02d}_target_{target_side}"
                _save_intervention_mri_panel(
                    (
                        target_volume[slice_index],
                        sample[f"{target_side}_t2"][0, slice_index].detach().cpu().numpy(),
                    ),
                    condition_images, condition_probabilities,
                    output_stem, target_side, reference, slice_index,
                )
        progress.update(index + 1, str(sample["patient_id"]))
    return records


def metric_bundle(frame: pd.DataFrame) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    pred = frame["prediction"].to_numpy(dtype=int)
    probs = frame[["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy()
    one_hot = np.eye(3)[y]
    result = {
        "n_hips": len(frame), "n_patients": frame["patient_id"].nunique(),
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "mae": mean_absolute_error(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "severe_error_rate": float(np.mean(np.abs(y - pred) == 2)),
    }
    try:
        result["macro_auroc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except ValueError:
        result["macro_auroc"] = np.nan
    for cls in range(3):
        mask = y == cls
        result[f"recall_class_{cls}"] = float(np.mean(pred[mask] == cls)) if mask.any() else np.nan
    return result


def aggregate_repeated_holdout(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment_id", "variant", "patient_id", "side", "label"]
    probabilities = ["prob_class_0", "prob_class_1", "prob_class_2"]
    aggregated = frame.groupby(keys, as_index=False)[probabilities].mean()
    aggregated["prediction"] = np.argmax(aggregated[probabilities].to_numpy(), axis=1)
    aggregated["confidence"] = aggregated[probabilities].max(axis=1)
    aggregated["abs_error"] = np.abs(aggregated["prediction"] - aggregated["label"])
    return aggregated


def paired_bootstrap(aggregated: pd.DataFrame, variant: str, n_bootstrap: int, seed: int) -> dict:
    full = aggregated[aggregated["variant"] == "full"]
    other = aggregated[aggregated["variant"] == variant]
    keys = ["patient_id", "side"]
    paired = full.merge(other, on=keys, suffixes=("_full", "_other"))
    if paired.empty:
        raise ValueError(f"full 与 {variant} 无成对预测")
    patients = paired["patient_id"].unique()
    rng = np.random.default_rng(seed)
    deltas = []
    progress = ProgressReporter(f"患者级 bootstrap：Full vs {variant}", n_bootstrap)
    for bootstrap_index in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([paired[paired["patient_id"] == patient] for patient in sampled], ignore_index=True)
        q_full = cohen_kappa_score(sample["label_full"], sample["prediction_full"], weights="quadratic")
        q_other = cohen_kappa_score(sample["label_other"], sample["prediction_other"], weights="quadratic")
        if np.isfinite(q_full) and np.isfinite(q_other):
            deltas.append(q_full - q_other)
        progress.update(bootstrap_index + 1)
    values = np.asarray(deltas)
    point = metric_bundle(full)["qwk"] - metric_bundle(other)["qwk"]
    p_value = 2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0)))
    return {
        "variant": variant, "delta_qwk": point,
        "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
        "p_raw": min(p_value, 1.0), "bootstrap_valid": len(values),
        "paired_patients": len(patients),
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p_values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def plot_analysis(metrics: pd.DataFrame, comparisons: pd.DataFrame, root: Path) -> None:
    fig, (ax1, ax2) = create_nature_figure("double", 105, ncols=2, gridspec_kw={"wspace": 0.55})
    order = comparisons.sort_values("delta_qwk").reset_index(drop=True)
    y = np.arange(len(order))
    xerr = np.vstack([order["delta_qwk"] - order["ci_low"], order["ci_high"] - order["delta_qwk"]])
    ax1.errorbar(order["delta_qwk"], y, xerr=xerr, fmt="o", color=COLORS["blue"],
                 ecolor=COLORS["black"], capsize=2, zorder=3)
    ax1.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax1.set_yticks(y, order["variant"])
    ax1.set_xlabel(r"$\Delta$QWK (Full − ablation)")
    clean_axis(ax1, grid=True)
    for i, row in order.iterrows():
        ax1.text(row["ci_high"], i, f"  $P_{{Holm}}$={row['p_holm']:.3g}", va="center", fontsize=6.5)

    fold_metrics = metrics[metrics["level"] == "fold"]
    variants = list(metrics[metrics["level"] == "aggregated"]["variant"])
    for i, variant in enumerate(variants):
        values = fold_metrics.loc[fold_metrics["variant"] == variant, "qwk"].to_numpy()
        color = COLORS["blue"] if variant == "full" else COLORS["gray"]
        marker = "o"
        ax2.scatter(np.full(len(values), i), values, facecolors=color if variant == "full" else "none",
                    edgecolors=color, marker=marker, linewidth=1, zorder=3)
        if len(values):
            ax2.hlines(np.mean(values), i - 0.25, i + 0.25, color=color, linewidth=1.5)
    ax2.set_xticks(range(len(variants)), variants, rotation=55, ha="right")
    ax2.set_ylabel("Quadratic weighted κ")
    clean_axis(ax2)
    ax1.text(-0.18, 1.03, "a", transform=ax1.transAxes, fontweight="bold")
    ax2.text(-0.18, 1.03, "b", transform=ax2.transAxes, fontweight="bold")
    save_publication_figure(fig, root / "figures" / "main" / "ablation_qwk")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_mechanisms(predictions: pd.DataFrame, root: Path) -> None:
    import matplotlib.pyplot as plt

    mechanism_path = root / "mechanisms" / "internal_mechanism_summary.csv"
    if not mechanism_path.exists():
        return
    mechanism = pd.read_csv(mechanism_path)
    full = mechanism[mechanism["variant"] == "full"].copy()
    if full.empty:
        return
    phenotypes = ["bilateral_normal", "unilateral", "concordant_bilateral", "discordant_bilateral"]
    fig, axes = create_nature_figure("double", 92, ncols=3, gridspec_kw={"wspace": 0.5})
    variables = ["modal_gate_left_mean", "left_diff_gate_mean", "left_diff_grad_activation"]
    labels = ["T1 modal gate", "Local difference gate", "Difference contribution"]
    rng = np.random.default_rng(42)
    for ax, variable, label in zip(axes, variables, labels):
        for index, phenotype in enumerate(phenotypes):
            values = full.loc[full["phenotype"] == phenotype, variable].dropna().to_numpy()
            if not len(values):
                continue
            jitter = rng.normal(index, 0.055, len(values))
            ax.scatter(jitter, values, s=8, facecolors="none", edgecolors=COLORS["gray"], linewidth=0.6)
            ax.hlines(np.median(values), index - 0.24, index + 0.24, color=COLORS["blue"], linewidth=1.5)
        ax.set_xticks(range(4), ["Normal", "Unilateral", "Concordant", "Discordant"],
                      rotation=35, ha="right")
        ax.set_ylabel(label)
        clean_axis(ax)
    for index, ax in enumerate(axes):
        ax.text(-0.18, 1.03, chr(97 + index), transform=ax.transAxes, fontweight="bold")
    save_publication_figure(fig, root / "figures" / "supplementary" / "mechanism_phenotypes")
    plt.close(fig)
    full.to_csv(root / "source_data" / "Fig_mechanism_phenotypes.csv", index=False)

    errors = predictions.groupby(["variant", "fold", "patient_id"], as_index=False)["abs_error"].mean()
    error_wide = errors.pivot_table(index=["fold", "patient_id"], columns="variant", values="abs_error").reset_index()
    if {"full", "no_registration"}.issubset(error_wide.columns):
        registration = full.groupby(["fold", "patient_id"], as_index=False)["ncc_improvement"].mean()
        relation = registration.merge(error_wide, on=["fold", "patient_id"])
        relation["error_improvement"] = relation["no_registration"] - relation["full"]
        fig, ax = create_nature_figure("single", 70)
        ax.scatter(relation["ncc_improvement"], relation["error_improvement"], s=9,
                   facecolors="none", edgecolors=COLORS["blue"], linewidth=0.7)
        if len(relation) >= 2:
            coefficient = np.polyfit(relation["ncc_improvement"], relation["error_improvement"], 1)
            x = np.linspace(relation["ncc_improvement"].min(), relation["ncc_improvement"].max(), 100)
            ax.plot(x, np.polyval(coefficient, x), color=COLORS["vermillion"])
        ax.axhline(0, color=COLORS["gray"], linestyle="--")
        ax.set_xlabel("Registration NCC improvement")
        ax.set_ylabel("Absolute-error improvement")
        clean_axis(ax)
        save_publication_figure(fig, root / "figures" / "supplementary" / "registration_error_relation")
        plt.close(fig)
        relation.to_csv(root / "source_data" / "Fig_registration_error_relation.csv", index=False)

def plot_contralateral_interventions(root: Path) -> None:
    import matplotlib.pyplot as plt

    intervention_path = root / "mechanisms" / "contralateral_interventions_full.csv"
    if not intervention_path.exists():
        print("[analyze] 未找到对侧干预 CSV，跳过患者级配对散点图和雨云图。")
        return

    interventions = pd.read_csv(intervention_path)
    required = {
        "fold", "patient_id", "target_side", "condition", "label",
        "reference_class_probability", "delta_reference_probability_vs_correct",
    }
    missing = required - set(interventions.columns)
    if missing:
        raise ValueError(f"对侧干预 CSV 缺少绘图所需字段: {sorted(missing)}")

    condition_order = [
        "random_wrong_patient", "stage_matched_wrong_patient", "zero", "self_copy",
    ]
    condition_labels = {
        "random_wrong_patient": "Random donor",
        "stage_matched_wrong_patient": "Stage-matched donor",
        "zero": "Zero input",
        "self_copy": "Self-copy",
    }
    available = set(interventions["condition"].unique())
    required_conditions = {"correct", *condition_order}
    if not required_conditions.issubset(available):
        missing_conditions = sorted(required_conditions - available)
        raise ValueError(f"对侧干预结果缺少预设条件: {missing_conditions}")

    patient_side = interventions.groupby(
        ["patient_id", "target_side", "condition"], as_index=False,
    ).agg(
        label=("label", "first"),
        reference_class_probability=("reference_class_probability", "mean"),
        delta_reference_probability_vs_correct=(
            "delta_reference_probability_vs_correct", "mean"),
        n_folds=("fold", "nunique"),
    )
    patient_level = patient_side.groupby(
        ["patient_id", "condition"], as_index=False,
    ).agg(
        reference_class_probability=("reference_class_probability", "mean"),
        delta_reference_probability_vs_correct=(
            "delta_reference_probability_vs_correct", "mean"),
        n_sides=("target_side", "nunique"),
    )

    patient_side.to_csv(
        root / "source_data" / "Fig_contralateral_interventions_patient_side.csv", index=False,
    )
    patient_level.to_csv(
        root / "source_data" / "Fig_contralateral_interventions_patient.csv", index=False,
    )

    probability_wide = patient_level.pivot(
        index="patient_id", columns="condition", values="reference_class_probability",
    )
    paired_fig, paired_axes = create_nature_figure(
        "double", 112, nrows=2, ncols=2,
        gridspec_kw={"wspace": 0.32, "hspace": 0.42},
    )
    paired_axes = np.asarray(paired_axes).ravel()
    for panel_index, (ax, condition) in enumerate(zip(paired_axes, condition_order)):
        paired = probability_wide[["correct", condition]].dropna()
        for correct_value, intervention_value in paired.to_numpy():
            ax.plot([0, 1], [correct_value, intervention_value],
                    color=COLORS["gray"], alpha=0.16, linewidth=0.55, zorder=1)
        ax.scatter(np.zeros(len(paired)), paired["correct"], s=9,
                   facecolors="white", edgecolors=COLORS["blue"], linewidth=0.55, zorder=2)
        ax.scatter(np.ones(len(paired)), paired[condition], s=9,
                   facecolors=COLORS["vermillion"], edgecolors=COLORS["vermillion"],
                   linewidth=0.4, alpha=0.75, zorder=2)
        ax.scatter([0, 1], [paired["correct"].median(), paired[condition].median()],
                   marker="D", s=24, color=COLORS["black"], zorder=4,
                   label="Median" if panel_index == 0 else None)
        ax.set_xticks([0, 1], ["Correct", condition_labels[condition]])
        ax.set_ylabel("Reference-stage probability")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"n={len(paired)} patients", fontsize=7, pad=3)
        clean_axis(ax)
        ax.text(-0.13, 1.04, chr(97 + panel_index), transform=ax.transAxes,
                fontweight="bold", fontsize=8)
    save_publication_figure(
        paired_fig,
        root / "figures" / "supplementary" / "contralateral_paired_patient_scatter",
    )
    plt.close(paired_fig)

    rain_fig, rain_ax = create_nature_figure("double", 82)
    rng = np.random.default_rng(42)
    summary_rows = []
    for position, condition in enumerate(condition_order):
        values = patient_level.loc[
            patient_level["condition"] == condition,
            "delta_reference_probability_vs_correct",
        ].dropna().to_numpy(dtype=float)
        if not len(values):
            continue

        violin = rain_ax.violinplot(
            [values], positions=[position], vert=False, widths=0.72,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS["sky"])
            body.set_edgecolor(COLORS["blue"])
            body.set_linewidth(0.7)
            body.set_alpha(0.42)
            vertices = body.get_paths()[0].vertices
            vertices[:, 1] = np.maximum(vertices[:, 1], position)

        rain_ax.boxplot(
            [values], positions=[position], vert=False, widths=0.16,
            showfliers=False, patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": COLORS["black"], "linewidth": 0.8},
            medianprops={"color": COLORS["vermillion"], "linewidth": 1.2},
            whiskerprops={"color": COLORS["black"], "linewidth": 0.7},
            capprops={"color": COLORS["black"], "linewidth": 0.7},
        )
        jitter = rng.normal(loc=position - 0.23, scale=0.035, size=len(values))
        rain_ax.scatter(values, jitter, s=8, facecolors="none", edgecolors=COLORS["gray"],
                        linewidth=0.5, alpha=0.58, zorder=2)
        summary_rows.append({
            "condition": condition,
            "n_patients": len(values),
            "mean_delta": float(np.mean(values)),
            "median_delta": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "negative_fraction": float(np.mean(values < 0)),
            "large_drop_fraction_0.05": float(np.mean(values < -0.05)),
            "large_drop_fraction_0.10": float(np.mean(values < -0.10)),
        })

    rain_ax.axvline(0, color=COLORS["black"], linestyle="--", linewidth=0.9, zorder=0)
    rain_ax.set_yticks(range(len(condition_order)), [condition_labels[c] for c in condition_order])
    rain_ax.set_xlabel(r"Patient-level change in reference-stage probability ($\Delta P$)")
    rain_ax.set_ylabel("")
    rain_ax.invert_yaxis()
    clean_axis(rain_ax, grid=True)
    save_publication_figure(
        rain_fig, root / "figures" / "supplementary" / "contralateral_patient_raincloud",
    )
    plt.close(rain_fig)
    pd.DataFrame(summary_rows).to_csv(
        root / "source_data" / "Fig_contralateral_interventions.csv", index=False,
    )


def analyze(spec: dict, n_bootstrap: int) -> None:
    root = ensure_layout(spec)
    print("[进度] 统计分析：读取并聚合内部预测")
    predictions = pd.read_csv(root / "predictions" / "internal_holdout_predictions.csv")
    aggregated = aggregate_repeated_holdout(predictions)
    aggregated.to_csv(root / "predictions" / "internal_patient_side_aggregated.csv", index=False)

    metric_rows = []
    for (variant, fold), frame in predictions.groupby(["variant", "fold"]):
        metric_rows.append({"level": "fold", "variant": variant, "fold": fold, **metric_bundle(frame)})
    for variant, frame in aggregated.groupby("variant"):
        metric_rows.append({"level": "aggregated", "variant": variant, "fold": np.nan, **metric_bundle(frame)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "metrics" / "ablation_metrics.csv", index=False)

    variants = [name for name in spec["variants"] if name != "full"]
    print(f"[进度] 统计分析：开始 {len(variants)} 项 Full 配对比较")
    comparisons = pd.DataFrame([
        paired_bootstrap(aggregated, name, n_bootstrap, int(spec["seed"]) + i)
        for i, name in enumerate(variants)
    ])
    comparisons["p_holm"] = holm_adjust(comparisons["p_raw"])
    comparisons.to_csv(root / "metrics" / "paired_bootstrap_full_vs_ablation.csv", index=False)
    metrics.to_csv(root / "source_data" / "Fig_ablation_qwk_metrics.csv", index=False)
    comparisons.to_csv(root / "source_data" / "Fig_ablation_qwk_comparisons.csv", index=False)
    plot_analysis(metrics, comparisons, root)
    plot_mechanisms(predictions, root)
    plot_contralateral_interventions(root)
    print("[进度] 统计分析：主图和机制图已生成，开始绘制混淆矩阵")

    import matplotlib.pyplot as plt
    confusion_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "confusion_gradient",
        ["#F8D8E6", "#C7BCD9", "#8FA4CF", "#6686BC", "#514A91"],
    )
    groups = list(aggregated.groupby("variant"))
    confusion_progress = ProgressReporter("绘制混淆矩阵", len(groups))
    for group_index, (variant, frame) in enumerate(groups):
        cm = confusion_matrix(frame["label"], frame["prediction"], labels=[0, 1, 2])
        percentage = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        fig, ax = create_nature_figure("single", 76)
        image = ax.imshow(percentage, cmap=confusion_cmap, vmin=0, vmax=100)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{percentage[row, col]:.1f}%\n({cm[row, col]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if percentage[row, col] > 55 else "black")
        ax.set_xticks(range(3), ["Normal", "Early", "Late"])
        ax.set_yticks(range(3), ["Normal", "Early", "Late"])
        ax.set_xlabel("Predicted stage"); ax.set_ylabel("Reference stage")
        fig.colorbar(image, ax=ax, label="Row percentage (%)", fraction=0.046)
        save_publication_figure(fig, root / "figures" / "supplementary" / f"confusion_{variant}")
        plt.close(fig)
        pd.DataFrame(cm).to_csv(root / "source_data" / f"confusion_{variant}.csv", index=False)
        confusion_progress.update(group_index + 1, variant)
    print(f"[analyze] 完成患者级聚合和 {n_bootstrap} 次成对 bootstrap。")


def infer_external(spec: dict, device_name: str, inference_batch_size: int = 1) -> None:
    root = ensure_layout(spec)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    data = load_manifest(spec["external_data_root"])
    ids = np.arange(len(data["patient_id"]), dtype=np.int64)
    dataset = ONFHDataset(spec["external_data_root"], ids,
                          volume_size=tuple(spec["training"]["volume_size"]), augment=False)
    loader = DataLoader(
        dataset, batch_size=inference_batch_size, shuffle=False,
        num_workers=int(spec["training"].get("num_workers", 0)),
        pin_memory=bool(spec["training"].get("pin_memory", False)),
    )
    probability_sums = {
        "left": np.zeros((len(dataset), 3), dtype=np.float64),
        "right": np.zeros((len(dataset), 3), dtype=np.float64),
    }
    folds = list(spec["folds"])
    overall = ProgressReporter("外部集成推理", len(folds))
    for fold_index, fold in enumerate(folds):
        model, _ = load_checkpoint(checkpoint_path(root, "full", fold), device)
        cursor = 0
        batch_progress = ProgressReporter(f"外部推理 Full / fold {fold}", len(loader))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            for batch_index, batch in enumerate(loader):
                inputs = [batch[key].to(device) for key in ("left_t1", "left_t2", "right_t1", "right_t2")]
                output = model(*inputs)
                batch_size = len(batch["patient_id"])
                for side in ("left", "right"):
                    probability_sums[side][cursor:cursor + batch_size] += (
                        ordinal_probs_from_logits(output[f"{side}_stage"], 3).cpu().numpy())
                cursor += batch_size
                del output, inputs
                batch_progress.update(batch_index + 1)
        del model
        clear_cuda_cache()
        overall.update(fold_index + 1, f"fold {fold}")

    rows = []
    for index, patient_id in enumerate(data["patient_id"]):
        for side in ("left", "right"):
            probability = probability_sums[side][index] / len(folds)
            label = int(data[f"label_{side}_ELS"][index])
            pred = int(np.argmax(probability))
            rows.append({"patient_id": str(patient_id), "side": side,
                         "label": label, "prediction": pred,
                         **{f"prob_class_{c}": float(probability[c]) for c in range(3)}})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "predictions" / "external_full_ensemble_predictions.csv", index=False)
    pd.DataFrame([metric_bundle(frame)]).to_csv(root / "metrics" / "external_full_metrics.csv", index=False)
    print(f"[infer-external] Full ensemble: {data['patient_id'].__len__()} patients, {len(frame)} hips.")


def build_case_manifest(spec: dict, output_path: Path) -> dict:
    case_root = Path(spec["case_data_root"])
    labels = load_manifest(spec["case_label_manifest"])
    label_index = {str(pid): i for i, pid in enumerate(labels["patient_id"])}
    pattern = re.compile(r"(mri\d+)", re.IGNORECASE)
    files = {}
    for modality in ("T1", "T2"):
        for side in ("Left", "Right"):
            for path in (case_root / modality / side).glob("*.nrrd"):
                match = pattern.search(path.name)
                if match:
                    files.setdefault(match.group(1), {})[f"{side.lower()}_{modality.lower()}_path"] = str(path)
    patients = sorted(pid for pid, paths in files.items() if len(paths) == 4)
    manifest = {key: [] for key in (
        "patient_id", "left_t1_path", "left_t2_path", "right_t1_path", "right_t2_path",
        "label_left_bin", "label_right_bin", "label_left_ELS", "label_right_ELS",
    )}
    for pid in patients:
        if pid not in label_index:
            raise KeyError(f"固定病例 {pid} 在标签 manifest 中不存在")
        i = label_index[pid]
        manifest["patient_id"].append(pid)
        for key in ("left_t1_path", "left_t2_path", "right_t1_path", "right_t2_path"):
            manifest[key].append(files[pid][key])
        for key in ("label_left_bin", "label_right_bin", "label_left_ELS", "label_right_ELS"):
            manifest[key].append(labels[key][i])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _normalize_map(image: np.ndarray, upper_quantile: float = 0.99) -> np.ndarray:
    image = np.nan_to_num(np.asarray(image, dtype=np.float32), copy=False)
    low = float(np.min(image))
    high = float(np.quantile(image, upper_quantile))
    if high <= low + 1e-8:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _right_cam_to_original_space(cam_volume: np.ndarray,
                                 displacement: np.ndarray | None,
                                 registration_type: str) -> np.ndarray:
    import torch.nn.functional as torch_f

    restored = np.asarray(cam_volume, dtype=np.float32)
    if registration_type == "deformable" and displacement is not None:
        cam = torch.from_numpy(restored).unsqueeze(0).unsqueeze(0)
        flow = torch.from_numpy(np.asarray(displacement, dtype=np.float32)).unsqueeze(0)
        _, _, depth, height, width = cam.shape
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth),
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )
        base_grid = torch.stack([x, y, z], dim=0).unsqueeze(0)
        inverse_grid = (base_grid - flow).permute(0, 2, 3, 4, 1)
        restored = torch_f.grid_sample(
            cam, inverse_grid, mode="bilinear", padding_mode="border",
            align_corners=True,
        )[0, 0].numpy()
    if registration_type in {"flip", "deformable"}:
        restored = np.flip(restored, axis=-1).copy()
    return restored


def _sacon_fusion_grad_cam(model: SACON, batch: dict, device: torch.device) -> dict:
    import torch.nn.functional as torch_f

    inputs = [
        batch[key].unsqueeze(0).to(device)
        for key in ("left_t1", "left_t2", "right_t1", "right_t2")
    ]
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        output = model(*inputs)
        probabilities = {
            side: ordinal_probs_from_logits(output[f"{side}_stage"], 3)
            for side in ("left", "right")
        }
        labels = {side: int(batch[f"{side}_stage"]) for side in ("left", "right")}
        score = sum(torch.log(probabilities[side][:, labels[side]] + 1e-8).sum()
                    for side in ("left", "right"))
        features = [output["z_left"], output["z_right"]]
        gradients = torch.autograd.grad(score, features, retain_graph=False, create_graph=False)

    spatial_size = tuple(inputs[0].shape[2:])
    result = {
        "left_t1": inputs[0][0, 0].detach().float().cpu().numpy(),
        "left_t2": inputs[1][0, 0].detach().float().cpu().numpy(),
        "right_t1": inputs[2][0, 0].detach().float().cpu().numpy(),
        "right_t2": inputs[3][0, 0].detach().float().cpu().numpy(),
    }
    for side, feature, gradient in zip(("left", "right"), features, gradients):
        feature32 = feature.detach().float()
        gradient32 = gradient.detach().float()
        weights = gradient32.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * feature32).sum(dim=1, keepdim=True))
        if float(cam.max()) <= 1e-12:
            cam = torch.relu(gradient32 * feature32).mean(dim=1, keepdim=True)
        fusion = feature32.abs().mean(dim=1, keepdim=True)
        result[f"{side}_cam"] = torch_f.interpolate(
            cam, size=spatial_size, mode="trilinear", align_corners=False,
        )[0, 0].cpu().numpy()
        result[f"{side}_fusion"] = torch_f.interpolate(
            fusion, size=spatial_size, mode="trilinear", align_corners=False,
        )[0, 0].cpu().numpy()
        result[f"{side}_probability"] = probabilities[side][0].detach().float().cpu().numpy()

    difference = output.get("diff_abs")
    result["difference"] = (
        np.zeros(spatial_size, dtype=np.float32)
        if difference is None else
        torch_f.interpolate(
            difference.detach().float().mean(1, keepdim=True), size=spatial_size,
            mode="trilinear", align_corners=False,
        )[0, 0].cpu().numpy()
    )
    displacement = output.get("disp_fields")
    result["displacement"] = (
        None if displacement is None else displacement[0].detach().float().cpu().numpy()
    )
    result["registration_type"] = model.config.registration_type
    del output, inputs, features, gradients
    return result


def visualize_cases(spec: dict, device_name: str) -> None:
    import matplotlib.pyplot as plt

    root = ensure_layout(spec)
    manifest_path = root / "diagnostics" / "fixed_case_manifest.json"
    manifest = build_case_manifest(spec, manifest_path)
    if len(manifest["patient_id"]) != 5:
        raise ValueError(f"预期固定 5 例，实际找到 {len(manifest['patient_id'])} 例")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    dataset = ONFHDataset(str(manifest_path), np.arange(5),
                          volume_size=tuple(spec["training"]["volume_size"]), augment=False)

    prediction_path = root / "predictions" / "internal_holdout_predictions.csv"
    mechanism_fold = int(spec["folds"][0])
    if prediction_path.exists():
        pred = pd.read_csv(prediction_path)
        full = pred[pred["variant"] == "full"]
        scores = {int(fold): metric_bundle(frame)["qwk"] for fold, frame in full.groupby("fold")}
        if scores:
            mechanism_fold = max(scores, key=scores.get)

    folds = list(spec["folds"])
    variants = list(spec["variants"])
    cam_checkpoints = resolve_checkpoint_matrix(root, variants, [mechanism_fold])
    probability_sums = {
        "left": np.zeros((5, 3), dtype=np.float64),
        "right": np.zeros((5, 3), dtype=np.float64),
    }

    ensemble_progress = ProgressReporter("固定五病例 Full 集成概率", len(folds))
    for fold_index, fold in enumerate(folds):
        model, _ = load_checkpoint(checkpoint_path(root, "full", fold), device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            for case_index in range(5):
                batch = dataset[case_index]
                inputs = [batch[key].unsqueeze(0).to(device)
                          for key in ("left_t1", "left_t2", "right_t1", "right_t2")]
                output = model(*inputs)
                for side in ("left", "right"):
                    probability_sums[side][case_index] += (
                        ordinal_probs_from_logits(output[f"{side}_stage"], 3)[0].cpu().numpy())
                del output, inputs
        del model
        clear_cuda_cache()
        ensemble_progress.update(fold_index + 1, f"fold {fold}")

    variant_maps: dict[str, list[dict]] = {variant: [] for variant in variants}
    cam_progress = ProgressReporter("全部消融融合特征 CAM", len(variants) * 5)
    completed = 0
    for variant in variants:
        model, _ = load_checkpoint(cam_checkpoints[(variant, mechanism_fold)], device)
        for case_index in range(5):
            variant_maps[variant].append(
                _sacon_fusion_grad_cam(model, dataset[case_index], device)
            )
            completed += 1
            cam_progress.update(completed, f"{variant} / Case {case_index + 1}")
        del model
        clear_cuda_cache()

    locked_slices = []
    for case_index in range(5):
        full_map = variant_maps["full"][case_index]
        depth_score = full_map["difference"].mean(axis=(1, 2))
        if not np.isfinite(depth_score).any() or float(np.nanmax(depth_score)) <= 1e-12:
            depth_score = (full_map["left_fusion"] + full_map["right_fusion"]).mean(axis=(1, 2))
        locked_slices.append(int(np.nanargmax(depth_score)))

    records = []
    for case_index in range(5):
        batch = dataset[case_index]
        for side in ("left", "right"):
            probability = probability_sums[side][case_index] / len(folds)
            records.append({
                "case": f"Case {case_index + 1}", "side": side,
                "variant": "full_ensemble", "label": int(batch[f"{side}_stage"]),
                "prediction": int(np.argmax(probability)), "mechanism_fold": mechanism_fold,
                "locked_slice": locked_slices[case_index],
                "cam_target": "not_applicable", "cam_base": "not_applicable",
                "cam_displayed": False,
                "cam_normalization": "within-map_99th-percentile",
                **{f"prob_class_{c}": float(probability[c]) for c in range(3)},
            })
        for variant in variants:
            for side in ("left", "right"):
                probability = variant_maps[variant][case_index][f"{side}_probability"]
                records.append({
                    "case": f"Case {case_index + 1}", "side": side, "variant": variant,
                    "label": int(batch[f"{side}_stage"]),
                    "prediction": int(np.argmax(probability)), "mechanism_fold": mechanism_fold,
                    "locked_slice": locked_slices[case_index],
                    "cam_target": "reference_stage",
                    "cam_base": (
                        "right_original_t1_left_hip_orientation"
                        if side == "right" else "left_original_t1"
                    ),
                    "cam_displayed": True,
                    "cam_normalization": "within-map_99th-percentile",
                    **{f"prob_class_{c}": float(probability[c]) for c in range(3)},
                })

    all_diff = np.concatenate([
        variant_maps["full"][i]["difference"].ravel() for i in range(5)
    ])
    diff_vmin, diff_vmax = np.quantile(all_diff, [0.01, 0.99])
    display_names = {
        "full": "Full", "no_contralateral": "No\ncontra.",
        "bilateral_no_diff": "No explicit\ndiff.", "no_registration": "No\nregistration",
        "flip_only": "Flip\nonly", "no_modal_gate": "No modal\ngate",
        "no_diff_attention": "No diff.\nattention", "no_decoupling": "No\ndecoupling",
        "no_local_diff_gate": "No local-\ndiff gate",
    }
    stage_names = ("Normal", "Early", "Late")

    def draw_case_information(ax, case_index: int) -> None:
        batch = dataset[case_index]
        left_stage = int(batch["left_stage"])
        right_stage = int(batch["right_stage"])
        patient_id = str(manifest["patient_id"][case_index])
        ax.set_facecolor("white")
        ax.text(0.02, 0.72, f"MRI ID\n{patient_id}", transform=ax.transAxes,
                fontsize=6.3, fontweight="bold", ha="left", va="top", color="black")
        ax.text(0.02, 0.43, f"Left:  {stage_names[left_stage]} ({left_stage})",
                transform=ax.transAxes, fontsize=6.1, ha="left", va="top", color="black")
        ax.text(0.02, 0.25, f"Right: {stage_names[right_stage]} ({right_stage})",
                transform=ax.transAxes, fontsize=6.1, ha="left", va="top", color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def clean_image_axis(ax) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def add_side_marker(ax, marker: str) -> None:
        ax.text(
            0.035, 0.955, marker, transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, fontweight="bold", color="white",
        )

    def add_modality_marker(ax, modality: str) -> None:
        ax.text(
            0.035, 0.045, modality, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.0, fontweight="bold", color="white",
        )

    difference_titles = ["Case\ninformation", "Left T1", "Left T2",
                         "Right T1", "Right T2", "Full\n|difference|"]
    difference_fig, difference_axes = create_nature_figure(
        "double", 145, nrows=5, ncols=len(difference_titles),
        gridspec_kw={"width_ratios": [1.35, 1, 1, 1, 1, 1],
                     "wspace": 0.035, "hspace": 0.075},
    )
    difference_axes = np.asarray(difference_axes)
    for row in range(5):
        slice_index = locked_slices[row]
        full_map = variant_maps["full"][row]
        left = full_map["left_t1"][slice_index]
        left_t2 = full_map["left_t2"][slice_index]
        right = full_map["right_t1"][slice_index]
        right_t2 = full_map["right_t2"][slice_index]
        t1_vmin, t1_vmax = _robust_image_limits([left, right])
        t2_vmin, t2_vmax = _robust_image_limits([left_t2, right_t2])
        draw_case_information(difference_axes[row, 0], row)
        difference_axes[row, 1].imshow(left, cmap="gray", vmin=t1_vmin, vmax=t1_vmax)
        difference_axes[row, 2].imshow(left_t2, cmap="gray", vmin=t2_vmin, vmax=t2_vmax)
        difference_axes[row, 3].imshow(right, cmap="gray", vmin=t1_vmin, vmax=t1_vmax)
        difference_axes[row, 4].imshow(right_t2, cmap="gray", vmin=t2_vmin, vmax=t2_vmax)
        difference_axes[row, 5].imshow(
            full_map["difference"][slice_index], cmap="magma",
            vmin=diff_vmin, vmax=max(diff_vmax, diff_vmin + 1e-8),
        )
        add_side_marker(difference_axes[row, 1], "L")
        add_side_marker(difference_axes[row, 2], "L")
        add_side_marker(difference_axes[row, 3], "R")
        add_side_marker(difference_axes[row, 4], "R")
        for ax in difference_axes[row, 1:]:
            clean_image_axis(ax)
    for column, title in enumerate(difference_titles):
        difference_axes[0, column].set_title(title, fontsize=7, pad=3)
    save_publication_figure(
        difference_fig, root / "figures" / "main" / "fixed_five_cases_difference",
    )
    plt.close(difference_fig)

    cam_titles = ["Case / side", "T1", "T2"] + [
        display_names.get(v, v) for v in variants
    ]
    cam_fig, cam_axes = create_nature_figure(
        "double", 220, nrows=10, ncols=len(cam_titles),
        gridspec_kw={"width_ratios": [1.35] + [1] * (2 + len(variants)),
                     "wspace": 0.025, "hspace": 0.055},
    )
    cam_axes = np.asarray(cam_axes)
    for case_index in range(5):
        slice_index = locked_slices[case_index]
        full_map = variant_maps["full"][case_index]
        left_t1 = full_map["left_t1"][slice_index]
        left_t2 = full_map["left_t2"][slice_index]
        right_t1 = full_map["right_t1"][slice_index]
        right_t2 = full_map["right_t2"][slice_index]
        t1_vmin, t1_vmax = _robust_image_limits([left_t1, right_t1])
        t2_vmin, t2_vmax = _robust_image_limits([left_t2, right_t2])
        for side_offset, side in enumerate(("left", "right")):
            row = 2 * case_index + side_offset
            stage = int(dataset[case_index][f"{side}_stage"])
            patient_id = str(manifest["patient_id"][case_index])
            cam_axes[row, 0].text(
                0.02, 0.68, f"Case {case_index + 1} | {side.title()}\nMRI ID {patient_id}\n"
                f"{stage_names[stage]} ({stage})", transform=cam_axes[row, 0].transAxes,
                fontsize=5.8, ha="left", va="top", color="black",
            )
            clean_image_axis(cam_axes[row, 0])

            t1 = left_t1 if side == "left" else np.flip(right_t1, axis=-1)
            t2 = left_t2 if side == "left" else np.flip(right_t2, axis=-1)
            cam_axes[row, 1].imshow(t1, cmap="gray", vmin=t1_vmin, vmax=t1_vmax)
            cam_axes[row, 2].imshow(t2, cmap="gray", vmin=t2_vmin, vmax=t2_vmax)
            add_side_marker(cam_axes[row, 1], "L" if side == "left" else "R")
            add_side_marker(cam_axes[row, 2], "L" if side == "left" else "R")
            add_modality_marker(cam_axes[row, 1], "T1")
            add_modality_marker(cam_axes[row, 2], "T2")
            clean_image_axis(cam_axes[row, 1])
            clean_image_axis(cam_axes[row, 2])

            for column, variant in enumerate(variants, start=3):
                maps = variant_maps[variant][case_index]
                cam_volume = maps["left_cam"]
                if side == "right":
                    cam_volume = _right_cam_to_original_space(
                        maps["right_cam"], maps["displacement"], maps["registration_type"],
                    )
                    cam_volume = np.flip(cam_volume, axis=-1)
                cam = _normalize_map(cam_volume[slice_index])
                cam_axes[row, column].imshow(t1, cmap="gray", vmin=t1_vmin, vmax=t1_vmax)
                cam_image = cam_axes[row, column].imshow(
                    cam, cmap="jet", vmin=0, vmax=1,
                    alpha=np.clip(0.6 * cam, 0.0, 0.6),
                )
                clean_image_axis(cam_axes[row, column])
    for column, title in enumerate(cam_titles):
        cam_axes[0, column].set_title(title, fontsize=6.3, pad=3)
    cam_fig.colorbar(
        cam_image, ax=cam_axes[:, 3:].ravel().tolist(), orientation="horizontal",
        fraction=0.018, pad=0.018, aspect=45, label="Normalized CAM",
    )
    save_publication_figure(cam_fig, root / "figures" / "main" / "fixed_five_cases_cam")
    plt.close(cam_fig)

    source_frame = pd.DataFrame(records)
    source_frame.to_csv(root / "source_data" / "Fig_fixed_five_cases.csv", index=False)
    source_frame[source_frame["variant"] == "full_ensemble"].to_csv(
        root / "source_data" / "Fig_fixed_five_cases_difference.csv", index=False,
    )
    source_frame[source_frame["cam_displayed"]].to_csv(
        root / "source_data" / "Fig_fixed_five_cases_cam.csv", index=False,
    )
    print(
        f"[visualize-mri] 已分别生成对侧差异面板和双侧 T1-CAM 面板；"
        f"5 例 × 双侧 × {len(variants)} 个结构均使用 fold {mechanism_fold} 和同一锁定切片。"
    )


def visualize_contralateral_mri(spec: dict, folds: tuple[int, ...], device_name: str,
                                max_visual_cases: int) -> None:
    if max_visual_cases <= 0:
        print("[visualize-mri] intervention_visualization_cases=0，跳过对侧干预 MRI 面板。")
        return
    if not folds:
        raise ValueError("visualize-mri 至少需要在 IDE_RUN_CONFIG.folds 中指定一个 fold。")

    root = ensure_layout(spec)
    data = load_manifest(spec["data_root"])
    splits = canonical_splits(spec, data)
    fold = int(folds[0])
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    path = checkpoint_path(root, "full", fold)
    model, _ = load_checkpoint(path, device)
    dataset = ONFHDataset(
        spec["data_root"], splits[fold][1],
        volume_size=tuple(spec["training"]["volume_size"]), augment=False,
    )
    figure_dir = root / "figures" / "supplementary" / "contralateral_interventions"
    try:
        visual_records = contralateral_interventions(
            model, dataset, device, fold, int(spec["seed"]),
            figure_dir=figure_dir,
            max_visual_cases=max_visual_cases,
            max_cases=max_visual_cases,
        )
        pd.DataFrame(visual_records).to_csv(
            root / "source_data" / "Fig_contralateral_intervention_cases.csv", index=False,
        )
    finally:
        del model, dataset
        clear_cuda_cache()
    print(
        f"[visualize-mri] 已使用 Full / fold {fold} 生成前 "
        f"{min(max_visual_cases, len(splits[fold][1]))} 例对侧干预 T1/T2 面板。"
    )


def visualize_mri(spec: dict, folds: tuple[int, ...], device_name: str,
                  intervention_cases: int = 5) -> None:
    print("[visualize-mri] 阶段 1/2：生成对侧干预 T1/T2 面板。", flush=True)
    visualize_contralateral_mri(spec, folds, device_name, intervention_cases)
    print("[visualize-mri] 阶段 2/2：生成固定五病例及全部消融融合特征 CAM。", flush=True)
    visualize_cases(spec, device_name)
    print("[visualize-mri] 全部 MRI 切片与类激活图已生成。", flush=True)


def audit_figures(spec: dict) -> None:
    root = ensure_layout(spec)
    rows = []
    files = [path for path in sorted((root / "figures").rglob("*")) if path.is_file()]
    progress = ProgressReporter("审计投稿图件", len(files))
    for index, path in enumerate(files):
        issues = audit_figure(path)
        rows.append({"path": str(path), "status": "PASS" if not issues else "FAIL",
                     "issues": "; ".join(issues)})
        progress.update(index + 1, path.name)
    report = pd.DataFrame(rows)
    report.to_csv(root / "diagnostics" / "figure_audit.csv", index=False)
    legends = (
        "Ablation effects are reported as the difference in quadratic weighted kappa between "
        "Full SACON and each ablated model. Points denote estimates and horizontal lines denote "
        "patient-level paired bootstrap 95% confidence intervals. P values are two-sided and "
        "Holm-adjusted across all prespecified comparisons. Repeated holdout probabilities were "
        "averaged for each patient-side before inference. One fixed-case panel shows bilateral T1/T2 "
        "and the Full contralateral difference; a separate panel shows reference-stage CAMs for all "
        "prespecified variants, transparently overlaid on the original right-hip T1. MRI identifiers "
        "and reference stages are placed in separate white information cells. CAMs indicate model "
        "response, not lesion localization."
    )
    (root / "figure_legends.md").write_text(legends, encoding="utf-8")
    (root / "figure_manifest.json").write_text(report.to_json(orient="records", indent=2), encoding="utf-8")
    print(report.to_string(index=False) if len(report) else "[audit] 尚无图文件。")


def selected(values: list[str], available: list[str]) -> list[str]:
    result = available if "all" in values else values
    unknown = set(result) - set(available)
    if unknown:
        raise ValueError(f"未知变体: {sorted(unknown)}")
    return result


def main() -> None:
    configure_live_console_output()
    os.chdir(PROJECT_ROOT)
    run = IDE_RUN_CONFIG
    spec_path = str(run["spec_path"])
    spec = load_spec(spec_path)
    available_variants = spec["variants"]
    variants = selected(list(run["variants"]), available_variants)
    folds = tuple(int(fold) for fold in run["folds"])

    unknown_folds = set(folds) - set(spec["folds"])
    if unknown_folds:
        raise ValueError(f"IDE_RUN_CONFIG 包含未知 fold: {sorted(unknown_folds)}")

    actions = {
        "prepare": lambda: prepare(spec, spec_path),
        "dry-run": lambda: dry_run(spec),
        "train": lambda: train_variants(spec, variants, folds, bool(run.get("resume", False))),
        "infer-internal": lambda: infer_internal(
            spec, variants, folds, str(run["device"]),
            inference_batch_size=int(run.get("inference_batch_size", 1)),
            enable_gradient_attribution=bool(run.get("enable_gradient_attribution", False)),
            run_interventions=bool(run.get("run_contralateral_interventions", True)),
        ),
        "analyze": lambda: analyze(spec, int(run["n_bootstrap"])),
        "visualize-mri": lambda: visualize_mri(
            spec, folds, str(run["device"]),
            intervention_cases=int(run.get("intervention_visualization_cases", 5)),
        ),
        "infer-external": lambda: infer_external(
            spec, str(run["device"]), int(run.get("inference_batch_size", 1))),
        "audit-figures": lambda: audit_figures(spec),
    }
    workflow_progress = ProgressReporter("整体流程", len(run["steps"]), updates=len(run["steps"]))
    for step_index, step in enumerate(run["steps"]):
        if step not in actions:
            raise ValueError(f"未知运行步骤: {step}；可选值为 {list(actions)}")
        print(f"\n[IDE] 开始执行: {step}", flush=True)
        started = time.perf_counter()
        actions[step]()
        elapsed = time.perf_counter() - started
        workflow_progress.update(step_index + 1, f"{step} 完成，用时 {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
