
import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AblationConfig
from data.dataset import ONFHDataset
from models.sacon import SACON
from utils.visualization import (
    DiffMapVisualizer,
    GradCAM3D,
    GradCAMPlusPlus3D,
    compare_bilateral_cam,
    save_cam_slices,
    save_diff_map_slices,
)


def ordinal_probs_from_logits(logits: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    cum_probs = torch.sigmoid(logits)
    if num_classes != cum_probs.size(1) + 1:
        raise ValueError(
            f"Expected {num_classes - 1} ordinal outputs, got {cum_probs.size(1)}."
        )

    probs = torch.zeros((cum_probs.size(0), num_classes), device=cum_probs.device)
    probs[:, 0] = 1.0 - cum_probs[:, 0]
    for class_index in range(1, num_classes - 1):
        probs[:, class_index] = cum_probs[:, class_index - 1] - cum_probs[:, class_index]
    probs[:, -1] = cum_probs[:, -1]
    probs = probs.clamp_min(0.0)
    return probs / (probs.sum(dim=1, keepdim=True) + 1e-8)


def safe_file_id(value: object) -> str:
    file_id = re.sub(r'[<>:"/\\|?*\s]+', "_", str(value)).strip("._")
    return (file_id or "unknown")[:120]


def format_stage_text(
    side: str,
    modality: str,
    probabilities: Sequence[float],
    prediction: int,
    true_label: int,
    class_names: Sequence[str],
) -> str:
    probability_text = ", ".join(
        f"{class_names[index]}={probability:.4f}"
        for index, probability in enumerate(probabilities)
    )
    return (
        f"{side}/{modality} | Probabilities: [{probability_text}] | "
        f"Pred: {class_names[prediction]} ({prediction}) | "
        f"True: {class_names[true_label]} ({true_label})"
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SACON, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "The selected file is not a complete best checkpoint: "
            "'model_state_dict' is missing."
        )

    train_cfg = dict(checkpoint.get("train_cfg", {}))
    checkpoint_task = train_cfg.get("task", "stage")
    if checkpoint_task != "stage":
        raise ValueError(
            f"This script only supports stage checkpoints, got task={checkpoint_task!r}."
        )

    ablation_cfg = AblationConfig(**checkpoint.get("ablation_cfg", {}))
    base_channels = int(train_cfg.get("base_channels", 32))
    num_stages = int(train_cfg.get("num_stages", 3))
    ordinal = bool(train_cfg.get("ordinal", True))
    if not ordinal:
        raise ValueError("This stage CAM script expects an ordinal stage checkpoint.")

    model = SACON(
        ablation_cfg,
        base_channels=base_channels,
        num_stages=num_stages,
        ordinal=True,
        task="stage",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"[Load] checkpoint={checkpoint_path} | "
        f"epoch={checkpoint.get('epoch', '?')} | num_stages={num_stages}"
    )
    return model, checkpoint


def test_indices_from_split_csv(split_csv: Path, fold: int) -> List[int]:
    indices: List[int] = []
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_fold = int(row.get("fold", fold))
            if row_fold == fold and row.get("split", "").lower() == "test":
                indices.append(int(row["source_index"]))
    if not indices:
        raise ValueError(f"No test rows for fold {fold} in {split_csv}.")
    return indices


def resolve_fold_test_indices(
    data: Dict[str, List],
    checkpoint_path: str,
    checkpoint: Dict[str, object],
    fold: int,
    split_csv: Optional[str],
) -> Tuple[np.ndarray, Optional[Path]]:
    candidates: List[Path] = []
    if split_csv:
        explicit_split = Path(split_csv)
        if not explicit_split.is_file():
            raise FileNotFoundError(f"The requested split CSV does not exist: {explicit_split}")
        candidates.append(explicit_split)
    else:
        checkpoint_dir = Path(checkpoint_path).resolve().parent
        candidates.extend(
            [
                checkpoint_dir / "fold_data_split.csv",
                checkpoint_dir.parent / f"fold_{fold}" / "fold_data_split.csv",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            indices = test_indices_from_split_csv(candidate, fold)
            print(f"[Split] Using saved fold table: {candidate}")
            return np.asarray(indices, dtype=np.int64), candidate

    from train import build_cv_splits

    train_cfg = dict(checkpoint.get("train_cfg", {}))
    splits = build_cv_splits(
        data,
        "stage",
        bool(train_cfg.get("group_by_base_id", False)),
        bool(train_cfg.get("repeated_holdout_7_3", False)),
        bool(train_cfg.get("real_only_test", False)),
        float(train_cfg.get("real_test_ratio", 0.3)),
    )
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {fold}.")
    print("[Split] No saved fold_data_split.csv found; reconstructed from checkpoint settings.")
    return np.asarray(splits[fold][1], dtype=np.int64), None


def resolve_fold_number(checkpoint_path: str, requested_fold: Optional[int]) -> int:
    if requested_fold is not None:
        return requested_fold
    match = re.search(r"fold[_-](\d+)", str(checkpoint_path), flags=re.IGNORECASE)
    if match:
        fold = int(match.group(1))
        print(f"[Fold] Inferred fold={fold} from the checkpoint path.")
        return fold
    print("[Fold] Could not infer a fold number from the checkpoint path; using fold=0.")
    return 0


def inference_with_cam(
    model: SACON,
    left_t1: torch.Tensor,
    left_t2: torch.Tensor,
    right_t1: torch.Tensor,
    right_t2: torch.Tensor,
    left_true: int,
    right_true: int,
    device: torch.device,
    out_dir: str,
    subject_id: str,
    class_names: Sequence[str],
    cam_type: str = "gradcam",
    dpi: int = 600,
) -> Dict[str, object]:
    subject_dir = Path(out_dir) / safe_file_id(subject_id)
    subject_dir.mkdir(parents=True, exist_ok=True)

    tensors = [left_t1, left_t2, right_t1, right_t2]
    left_t1, left_t2, right_t1, right_t2 = [tensor.to(device) for tensor in tensors]

    model.eval()
    with torch.no_grad():
        outputs = model(left_t1, left_t2, right_t1, right_t2)

    num_classes = len(class_names)
    left_probs_tensor = ordinal_probs_from_logits(outputs["left_stage"], num_classes)
    right_probs_tensor = ordinal_probs_from_logits(outputs["right_stage"], num_classes)
    left_prediction = int(left_probs_tensor.argmax(dim=1).item())
    right_prediction = int(right_probs_tensor.argmax(dim=1).item())
    left_probs = left_probs_tensor[0].detach().cpu().tolist()
    right_probs = right_probs_tensor[0].detach().cpu().tolist()

    cam_class = GradCAMPlusPlus3D if cam_type == "gradcam++" else GradCAM3D
    try:
        left_generator = cam_class(model, target_layer_name="left_stage_head_input")
        left_cam = left_generator.generate(
            left_t1,
            left_t2,
            right_t1,
            right_t2,
            target="left_stage",
            class_idx=left_prediction,
        )[0]

        right_generator = cam_class(model, target_layer_name="right_stage_head_input")
        right_cam = right_generator.generate(
            left_t1,
            left_t2,
            right_t1,
            right_t2,
            target="right_stage",
            class_idx=right_prediction,
        )[0]
    finally:
        model.remove_cam_hooks()

    images = {
        ("L", "T1"): left_t1.squeeze().detach().cpu().numpy(),
        ("L", "T2"): left_t2.squeeze().detach().cpu().numpy(),
        ("R", "T1"): np.flip(right_t1.squeeze().detach().cpu().numpy(), axis=2),
        ("R", "T2"): np.flip(right_t2.squeeze().detach().cpu().numpy(), axis=2),
    }
    side_values = {
        "L": (left_cam, left_probs, left_prediction, left_true),
        "R": (right_cam, right_probs, right_prediction, right_true),
    }

    cam_paths: Dict[str, str] = {}
    for (side, modality), image in images.items():
        cam, probabilities, prediction, true_label = side_values[side]
        path = subject_dir / f"cam_{side}_{modality}_{cam_type}.png"
        save_cam_slices(
            cam,
            str(path),
            image_3d=image,
            n_slices=19,
            title=(
                f"MRI ID: {subject_id} | "
                + format_stage_text(
                    side,
                    modality,
                    probabilities,
                    prediction,
                    true_label,
                    class_names,
                )
            ),
            dpi=dpi,
        )
        cam_paths[f"{side}_{modality}"] = str(path)

    bilateral_path = subject_dir / f"cam_bilateral_T1_{cam_type}.png"
    left_summary = format_stage_text(
        "L", "T1", left_probs, left_prediction, left_true, class_names
    )
    right_summary = format_stage_text(
        "R", "T1", right_probs, right_prediction, right_true, class_names
    )
    compare_bilateral_cam(
        left_cam,
        right_cam,
        str(bilateral_path),
        image_left=images[("L", "T1")],
        image_right=images[("R", "T1")],
        axis=0,
        n_slices=5,
        title=f"MRI ID: {subject_id}\n{left_summary}\n{right_summary}",
        dpi=dpi,
    )

    diff_path: Optional[Path] = None
    if outputs.get("diff_abs") is not None:
        diff_heatmap = DiffMapVisualizer.get_diff_heatmap(outputs["diff_abs"])[0]
        diff_path = subject_dir / "difference_response_map.png"
        save_diff_map_slices(
            diff_heatmap,
            str(diff_path),
            n_slices=5,
            title=f"MRI ID: {subject_id} | Difference Response Map |Delta F|",
            dpi=dpi,
        )

    result: Dict[str, object] = {
        "subject_id": subject_id,
        "left_stage_probs": left_probs,
        "right_stage_probs": right_probs,
        "left_prediction": left_prediction,
        "right_prediction": right_prediction,
        "left_true": left_true,
        "right_true": right_true,
        "cam_paths": cam_paths,
        "bilateral_path": str(bilateral_path),
        "diff_map_path": str(diff_path) if diff_path else None,
    }
    print(
        f"[Done] {subject_id} | "
        f"L pred/true={left_prediction}/{left_true} | "
        f"R pred/true={right_prediction}/{right_true} | {subject_dir}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SACON stage fold-test inference + CAM")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=r"G:\【重要】文章-汇总\【Article】SACON_-Symmetry-Aware Contralateral-difference Analysis-\SACON-原始结果\SACON-no-ssl-分期任务\stage_fold_0\stage_full_ssl_best.pth",
        help="Stage checkpoint for the selected fold.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Zero-based fold index; inferred from a fold_N checkpoint path when omitted.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS_Generation\50_7\siemens_onfh_data_generation.json",
        help="Dataset JSON; defaults to train_cfg.data_root stored in the checkpoint.",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="Optional fold_data_split.csv; auto-detected beside the checkpoint by default.",
    )
    parser.add_argument("--out_dir", type=str, default=r"E:\BackupToDriveF\【重要】文章-汇总\进行中-【Article】SACON_-Symmetry-Aware Contralateral-difference Analysis-\主要结果\实验结果图-性能评估-消融实验\源文件\SACON-NoSSL-CAM")
    parser.add_argument(
        "--cam_type", type=str, default="gradcam", choices=["gradcam", "gradcam++"]
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    fold = resolve_fold_number(args.ckpt, args.fold)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(args.ckpt, device)
    train_cfg = dict(checkpoint.get("train_cfg", {}))
    data_root = args.data_root or train_cfg.get("data_root")
    if not data_root:
        raise ValueError("data_root is absent from both --data_root and checkpoint train_cfg.")

    with open(data_root, "r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=OrderedDict)

    test_indices, _ = resolve_fold_test_indices(
        data=data,
        checkpoint_path=args.ckpt,
        checkpoint=checkpoint,
        fold=fold,
        split_csv=args.split_csv,
    )
    volume_size = tuple(train_cfg.get("volume_size", (19, 314, 224)))
    dataset = ONFHDataset(
        data_root=data_root,
        ids=test_indices,
        volume_size=volume_size,
        augment=False,
        online_train_augment=False,
    )

    num_stages = int(train_cfg.get("num_stages", 3))
    default_names = ("Normal", "Early", "Late")
    class_names = tuple(train_cfg.get("class_names", default_names))
    if len(class_names) != num_stages:
        class_names = tuple(f"Stage {index}" for index in range(num_stages))

    print(
        f"[Fold test] fold={fold} | samples={len(dataset)} | "
        f"device={device} | output={Path(args.out_dir).resolve()}"
    )
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        inference_with_cam(
            model=model,
            left_t1=sample["left_t1"].unsqueeze(0),
            left_t2=sample["left_t2"].unsqueeze(0),
            right_t1=sample["right_t1"].unsqueeze(0),
            right_t2=sample["right_t2"].unsqueeze(0),
            left_true=int(sample["label_left_ELS"].item()),
            right_true=int(sample["label_right_ELS"].item()),
            device=device,
            out_dir=args.out_dir,
            subject_id=str(sample["patient_id"]),
            class_names=class_names,
            cam_type=args.cam_type,
            dpi=600,
        )


if __name__ == "__main__":
    main()
