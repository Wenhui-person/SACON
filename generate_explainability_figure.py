

import os
import sys
import argparse
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from scipy import ndimage
import SimpleITK as sitk

from configs.config import AblationConfig
from models.sacon import SACON
from utils.visualization import DiffMapVisualizer


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def preprocess(volume: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)

    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 1e-6:
        volume = (volume - v_min) / (v_max - v_min)

    return volume.astype(np.float32)


def resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    target_d, target_h, target_w = target_size
    src_d, src_h, src_w = volume.shape
    zoom_factors = (target_d / src_d, target_h / src_h, target_w / src_w)
    return ndimage.zoom(volume, zoom_factors, order=1)


def load_volume(path: str, target_size: Tuple[int, int, int]) -> torch.Tensor:
    image = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(image)
    data = resize_volume(data, target_size)
    data = preprocess(data)
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)


def load_model_from_checkpoint(args, device: torch.device) -> SACON:
    ckpt = torch.load(args.ckpt, map_location=device)
    ablation_cfg = AblationConfig(**ckpt["ablation_cfg"])
    base_ch = ckpt.get("train_cfg", {}).get("base_channels", 32)
    num_stages = ckpt.get("train_cfg", {}).get("num_stages", 4)

    model = SACON(
        ablation_cfg,
        base_channels=base_ch,
        num_stages=num_stages,
        ordinal=True,
        task=args.task,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def ordinal_stage_probs(logits: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    cum_probs = torch.sigmoid(logits)
    probs = torch.zeros((cum_probs.size(0), num_classes), device=cum_probs.device)

    if num_classes != cum_probs.size(1) + 1:
        raise ValueError(
            f"Expected ordinal logits with num_classes-1 outputs, got {cum_probs.size(1)}."
        )

    probs[:, 0] = 1.0 - cum_probs[:, 0]
    for cls_idx in range(1, num_classes - 1):
        probs[:, cls_idx] = cum_probs[:, cls_idx - 1] - cum_probs[:, cls_idx]
    probs[:, -1] = cum_probs[:, -1]

    probs = probs.clamp_min(0.0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return probs


def ordinal_stage_scores(logits: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    cum_probs = torch.sigmoid(logits)
    scores = torch.zeros((cum_probs.size(0), num_classes), device=cum_probs.device)

    if num_classes != cum_probs.size(1) + 1:
        raise ValueError(
            f"Expected ordinal logits with num_classes-1 outputs, got {cum_probs.size(1)}."
        )

    scores[:, 0] = 1.0 - cum_probs[:, 0]
    for cls_idx in range(1, num_classes - 1):
        scores[:, cls_idx] = cum_probs[:, cls_idx - 1] - cum_probs[:, cls_idx]
    scores[:, -1] = cum_probs[:, -1]
    return scores


def predict_for_caption(outputs: Dict[str, torch.Tensor], task: str) -> Dict[str, float]:
    if task == "bin":
        left_prob = torch.sigmoid(outputs["left_binary"]).item()
        right_prob = torch.sigmoid(outputs["right_binary"]).item()
        return {
            "left_value": left_prob,
            "right_value": right_prob,
            "left_label": int(left_prob > 0.5),
            "right_label": int(right_prob > 0.5),
        }

    left_probs = ordinal_stage_probs(outputs["left_stage"], num_classes=3)
    right_probs = ordinal_stage_probs(outputs["right_stage"], num_classes=3)
    left_stage = int(left_probs.argmax(dim=1).item())
    right_stage = int(right_probs.argmax(dim=1).item())
    return {
        "left_value": float(left_probs.max().item()),
        "right_value": float(right_probs.max().item()),
        "left_label": left_stage,
        "right_label": right_stage,
    }


def generate_cams(
    model: SACON,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    task: str,
    cam_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    xl_t1, xl_t2, xr_t1, xr_t2 = inputs

    if task == "bin":
        left_target, right_target = "left_binary", "right_binary"
        left_module = model.left_bin_head.proj[0]
        right_module = model.right_bin_head.proj[0]
    else:
        left_target, right_target = "left_stage", "right_stage"
        left_module = model.left_stage_head.proj[0]
        right_module = model.right_stage_head.proj[0]

    cam_left = generate_module_cam(
        model=model,
        inputs=inputs,
        target_name=left_target,
        target_module=left_module,
        task=task,
        side="left",
        cam_type=cam_type,
        target_size=xl_t1.shape[2:],
    )[0]
    cam_right = generate_module_cam(
        model=model,
        inputs=inputs,
        target_name=right_target,
        target_module=right_module,
        task=task,
        side="right",
        cam_type=cam_type,
        target_size=xr_t1.shape[2:],
    )[0]
    return cam_left, cam_right


def generate_module_cam(
    model: SACON,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_name: str,
    target_module: torch.nn.Module,
    task: str,
    side: str,
    cam_type: str,
    target_size: Tuple[int, int, int],
) -> np.ndarray:
    feat_store: Dict[str, torch.Tensor] = {}
    grad_store: Dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        feat_store["value"] = out

    def bwd_hook(_module, _grad_in, grad_out):
        grad_store["value"] = grad_out[0] if grad_out else None

    h_fwd = target_module.register_forward_hook(fwd_hook)
    h_bwd = target_module.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        outputs = model(*inputs)
        score = select_cam_score(outputs[target_name], task=task, side=side)

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        feat_map = feat_store.get("value")
        gradient = grad_store.get("value")
        if feat_map is None:
            raise RuntimeError(f"Feature map was not captured for {side} {target_name}.")

        return cam_from_feature_map(feat_map, gradient, target_size, cam_type=cam_type)
    finally:
        h_fwd.remove()
        h_bwd.remove()


def select_cam_score(logits: torch.Tensor, task: str, side: str) -> torch.Tensor:
    del side
    if task == "bin":
        return logits.squeeze(1).sum()

    probs = ordinal_stage_scores(logits, num_classes=3)
    class_idx = int(probs.argmax(dim=1)[0].item())
    return probs[:, class_idx].sum()


def cam_from_feature_map(
    feat_map: torch.Tensor,
    gradient: Optional[torch.Tensor],
    target_size: Tuple[int, int, int],
    cam_type: str,
) -> np.ndarray:
    if gradient is None or gradient.abs().max() < 1e-8:
        raise RuntimeError("CAM gradient is empty; check target and computation graph.")

    if cam_type == "gradcam++":
        grad_sq = gradient ** 2
        grad_cu = gradient ** 3
        spatial_sum = feat_map.clamp_min(0).sum(dim=[2, 3, 4], keepdim=True)
        alpha = grad_sq / (2.0 * grad_sq + spatial_sum * grad_cu + 1e-8)
        weights = (alpha * torch.relu(gradient)).sum(dim=[2, 3, 4], keepdim=True)
    else:
        weights = gradient.mean(dim=[2, 3, 4], keepdim=True)

    cam = (weights * feat_map).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, size=target_size, mode="trilinear", align_corners=False
    )
    cam_np = cam.squeeze(1).detach().cpu().numpy()

    for i in range(cam_np.shape[0]):
        p_low, p_high = np.percentile(cam_np[i], [1, 99])
        if p_high > p_low:
            cam_np[i] = np.clip((cam_np[i] - p_low) / (p_high - p_low), 0, 1)
        else:
            vmin, vmax = cam_np[i].min(), cam_np[i].max()
            if vmax > vmin:
                cam_np[i] = (cam_np[i] - vmin) / (vmax - vmin)

    return cam_np


def compute_diff_heatmap(
    model: SACON,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Optional[np.ndarray]:
    with torch.no_grad():
        outputs = model(*inputs)
    if outputs.get("diff_abs") is None:
        return None
    diff = DiffMapVisualizer.get_diff_heatmap(outputs["diff_abs"])[0]
    return diff


def select_slice(cam: np.ndarray, axis: int, mode: str) -> int:
    if mode == "center":
        return cam.shape[axis] // 2

    projection_axes = tuple(i for i in range(3) if i != axis)
    scores = cam.mean(axis=projection_axes)
    return int(np.argmax(scores))


def extract_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return volume[idx, :, :]
    if axis == 1:
        return volume[:, idx, :]
    if axis == 2:
        return volume[:, :, idx]
    raise ValueError("axis must be 0, 1, or 2.")


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    return img


def overlay_cam(
    ax,
    image_slice: np.ndarray,
    cam_slice: np.ndarray,
    title: str,
    threshold: float,
    alpha: float,
) -> None:
    ax.imshow(normalize_image(image_slice), cmap="gray", interpolation="nearest")
    cam_show = np.where(cam_slice >= threshold, cam_slice, np.nan)
    ax.imshow(
        cam_show,
        cmap="inferno",
        vmin=0,
        vmax=1,
        alpha=alpha,
        interpolation="nearest",
    )
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_publication_figure(
    subject_id: str,
    task: str,
    pred: Dict[str, float],
    img_left: np.ndarray,
    img_right: np.ndarray,
    cam_left: np.ndarray,
    cam_right: np.ndarray,
    diff_heatmap: Optional[np.ndarray],
    out_dir: str,
    axis: int,
    slice_mode: str,
    threshold: float,
    alpha: float,
) -> Dict[str, str]:
    set_publication_style()
    os.makedirs(out_dir, exist_ok=True)

    idx_left = select_slice(cam_left, axis, slice_mode)
    idx_right = select_slice(cam_right, axis, slice_mode)
    idx_diff = select_slice(diff_heatmap, axis, slice_mode) if diff_heatmap is not None else idx_left

    left_img_sl = extract_slice(img_left, axis, idx_left)
    right_img_sl = extract_slice(img_right, axis, idx_right)
    left_cam_sl = extract_slice(cam_left, axis, idx_left)
    right_cam_sl = extract_slice(cam_right, axis, idx_right)

    fig = plt.figure(figsize=(7.0, 2.55))
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.08,
        left=0.035,
        right=0.96,
        bottom=0.12,
        top=0.83,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    if task == "bin":
        left_title = f"Left hip, P={pred['left_value']:.3f}"
        right_title = f"Right hip, P={pred['right_value']:.3f}"
        pred_text = (
            f"{subject_id} | Binary prediction: "
            f"L={pred['left_label']} ({pred['left_value']:.3f}), "
            f"R={pred['right_label']} ({pred['right_value']:.3f})"
        )
    else:
        left_title = f"Left hip, Stage {pred['left_label']} ({pred['left_value']:.3f})"
        right_title = f"Right hip, Stage {pred['right_label']} ({pred['right_value']:.3f})"
        pred_text = (
            f"{subject_id} | Stage prediction: "
            f"L=Stage {pred['left_label']} ({pred['left_value']:.3f}), "
            f"R=Stage {pred['right_label']} ({pred['right_value']:.3f})"
        )

    overlay_cam(axes[0], left_img_sl, left_cam_sl, left_title, threshold, alpha)
    overlay_cam(axes[1], right_img_sl, right_cam_sl, right_title, threshold, alpha)

    if diff_heatmap is not None:
        diff_sl = extract_slice(diff_heatmap, axis, idx_diff)
        axes[2].imshow(diff_sl, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(f"Difference response, idx={idx_diff}", pad=3)
    else:
        axes[2].text(
            0.5, 0.5, "Difference map\nnot available",
            ha="center", va="center", fontsize=8, color="#333333"
        )
        axes[2].set_title("Difference response", pad=3)

    axes[2].set_xticks([])
    axes[2].set_yticks([])
    for spine in axes[2].spines.values():
        spine.set_visible(False)

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(
            0.015, 0.985, label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.55, pad=1.2),
        )

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("CAM intensity", rotation=90, labelpad=6)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)

    fig.suptitle(pred_text, y=0.965, fontsize=8.5, fontweight="bold")

    base = os.path.join(out_dir, f"{subject_id}_explainability_panel")
    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SACON explainability panel.")
    parser.add_argument("--ckpt", type=str, default=r'D:\projects\PHD2\ONFH_SACON\checkpoints\stage_full_best_sacon.pth', help="Model checkpoint path.")
    parser.add_argument("--left_t1", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_2026126_Aug\test_data\T1\Left\t1_tse_cor_left_mri202613000007.nrrd", help="Left hip T1 NRRD path.")
    parser.add_argument("--left_t2", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_2026126_Aug\test_data\T2\Left\t2_tirm_cor_left_mri202613000007.nrrd",help="Left hip T2 NRRD path.")
    parser.add_argument("--right_t1", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_2026126_Aug\test_data\T1\Right\t1_tse_cor_right_mri202613000007.nrrd",help="Right hip T1 NRRD path.")
    parser.add_argument("--right_t2", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_2026126_Aug\test_data\T2\Right\t2_tirm_cor_right_mri202613000007.nrrd",help="Right hip T2 NRRD path.")
    parser.add_argument("--out_dir", type=str, default="outputs/explainability")
    parser.add_argument("--subject_id", type=str, default="mri202613000007")
    parser.add_argument("--task", type=str, default="stage", choices=["bin", "stage"])
    parser.add_argument("--cam_type", type=str, default="gradcam", choices=["gradcam", "gradcam++"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_size", type=int, nargs=3, default=[19, 314, 224])
    parser.add_argument("--axis", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--slice_mode", type=str, default="center", choices=["maxcam", "center"])
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument(
        "--flip_right_for_display",
        action="store_true",
        help="Flip right hip image and CAM on W axis for side-by-side display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args, device)

    target_size = tuple(args.target_size)
    xl_t1 = load_volume(args.left_t1, target_size).to(device)
    xl_t2 = load_volume(args.left_t2, target_size).to(device)
    xr_t1 = load_volume(args.right_t1, target_size).to(device)
    xr_t2 = load_volume(args.right_t2, target_size).to(device)
    inputs = (xl_t1, xl_t2, xr_t1, xr_t2)

    with torch.no_grad():
        outputs = model(*inputs)
    pred = predict_for_caption(outputs, args.task)

    cam_left, cam_right = generate_cams(model, inputs, args.task, args.cam_type)
    diff_heatmap = compute_diff_heatmap(model, inputs)

    img_left = xl_t1.squeeze().detach().cpu().numpy()
    img_right = xr_t1.squeeze().detach().cpu().numpy()

    if args.flip_right_for_display:
        img_right = np.flip(img_right, axis=2)
        cam_right = np.flip(cam_right, axis=2)

    paths = save_publication_figure(
        subject_id=args.subject_id,
        task=args.task,
        pred=pred,
        img_left=img_left,
        img_right=img_right,
        cam_left=cam_left,
        cam_right=cam_right,
        diff_heatmap=diff_heatmap,
        out_dir=args.out_dir,
        axis=args.axis,
        slice_mode=args.slice_mode,
        threshold=args.threshold,
        alpha=args.alpha,
    )

    print("[Explainability figure]")
    print(f"  PNG: {paths['png']}")
    print(f"  PDF: {paths['pdf']}")


if __name__ == "__main__":
    main()
