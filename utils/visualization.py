
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Dict, Tuple, List
from models.sacon import SACON

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})


class GradCAM3D:

    def __init__(self, model: SACON, target_layer_name: str = 't1_layer3'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.model.register_cam_hooks()

    def generate(
        self,
        xl_t1: torch.Tensor,
        xl_t2: torch.Tensor,
        xr_t1: torch.Tensor,
        xr_t2: torch.Tensor,
        target: str = 'right_stage',
        class_idx: Optional[int] = None,
        use_sigmoid: bool = True,
    ) -> np.ndarray:
        self.model.eval()
        self.model._feat_maps.clear()
        self.model._gradients.clear()

        xl_t1 = xl_t1.requires_grad_(True)
        outputs = self.model(xl_t1, xl_t2, xr_t1, xr_t2)

        logits = outputs[target]

        if target in ['left_stage', 'right_stage']:
            cum_probs = torch.sigmoid(logits)
            stage_probs = torch.zeros((cum_probs.size(0), cum_probs.size(1) + 1), device=cum_probs.device)

            stage_probs[:, 0] = 1.0 - cum_probs[:, 0]
            for i in range(1, stage_probs.size(1) - 1):
                stage_probs[:, i] = cum_probs[:, i - 1] - cum_probs[:, i]
            stage_probs[:, -1] = cum_probs[:, -1]

            stage_probs = stage_probs.clamp_min(0.0)
            stage_probs = stage_probs / (stage_probs.sum(dim=1, keepdim=True) + 1e-8)

            if class_idx is None:
                class_idx = stage_probs.argmax(dim=1)[0].item()

            score = stage_probs[:, int(class_idx)].sum()

        else:
            if logits.shape[-1] == 1 or (len(logits.shape) == 2 and logits.shape[1] == 1):
                if use_sigmoid:
                    score = torch.sigmoid(logits).squeeze(1).sum()
                else:
                    score = logits.squeeze(1).sum()
            else:
                if class_idx is None:
                    class_idx = logits.argmax(dim=1)[0].item()
                score = logits[:, int(class_idx)].sum()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        feat_map = self.model._feat_maps.get(self.target_layer_name)
        gradient = self.model._gradients.get(self.target_layer_name)


        print(f" Target layer: {self.target_layer_name}")
        if feat_map is not None:
            print(f"  Feature map shape: {feat_map.shape} | requires_grad: {feat_map.requires_grad}")
        if gradient is not None:
            print(
                f"  Gradient shape: {gradient.shape} | min={gradient.min():.6f}, max={gradient.max():.6f}, mean={gradient.mean():.6f}")
        else:
            print("  Gradient is still None !")

        if gradient is None or gradient.abs().max() < 1e-8:
            raise RuntimeError(
                f"目标层 '{self.target_layer_name}' 的梯度依然为0或None。\n"
                "可能原因：1. head内部有detach()；2. class_idx选择错误；3. head没有使用z_left。"
            )

        print(f"  Feature map shape: {feat_map.shape}")
        print(f"  Gradient shape: {gradient.shape}")
        print(f"  Gradient stats: min={gradient.min():.4f}, max={gradient.max():.4f}, mean={gradient.mean():.4f}")


        weights = gradient.mean(dim=[2, 3, 4], keepdim=True)
        print(f"  Weights stats: min={weights.min():.4f}, max={weights.max():.4f}")

        cam = (weights * feat_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        print(f"  CAM before interpolation: min={cam.min():.4f}, max={cam.max():.4f}")

        target_size = xl_t1.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode='trilinear', align_corners=False)

        cam = cam.squeeze(1).detach().cpu().numpy()

        for i in range(cam.shape[0]):
            p_low = np.percentile(cam[i], 1)
            p_high = np.percentile(cam[i], 99)

            if p_high > p_low:
                cam[i] = np.clip((cam[i] - p_low) / (p_high - p_low), 0, 1)
            else:
                vmin, vmax = cam[i].min(), cam[i].max()
                if vmax > vmin:
                    cam[i] = (cam[i] - vmin) / (vmax - vmin)

            print(f"  CAM[{i}] after norm: min={cam[i].min():.4f}, max={cam[i].max():.4f}, "
                  f"mean={cam[i].mean():.4f}, non-zero ratio={np.count_nonzero(cam[i]) / cam[i].size:.2%}")

        return cam

    def __del__(self):
        try:
            self.model.remove_cam_hooks()
        except Exception:
            pass


class GradCAMPlusPlus3D(GradCAM3D):

    def generate(
            self,
            xl_t1: torch.Tensor,
            xl_t2: torch.Tensor,
            xr_t1: torch.Tensor,
            xr_t2: torch.Tensor,
            target: str = 'right_stage',
            class_idx: Optional[int] = None,
            use_sigmoid: bool = True,
    ) -> np.ndarray:
        self.model.eval()
        self.model._feat_maps.clear()
        self.model._gradients.clear()

        xl_t1 = xl_t1.requires_grad_(True)
        xl_t2 = xl_t2.requires_grad_(True)
        xr_t1 = xr_t1.requires_grad_(True)
        xr_t2 = xr_t2.requires_grad_(True)

        outputs = self.model(xl_t1, xl_t2, xr_t1, xr_t2)
        logits = outputs[target]

        if target in ['left_stage', 'right_stage']:
            cum_probs = torch.sigmoid(logits)
            stage_probs = torch.zeros(
                (cum_probs.size(0), cum_probs.size(1) + 1),
                device=cum_probs.device,
            )
            stage_probs[:, 0] = 1.0 - cum_probs[:, 0]
            for i in range(1, stage_probs.size(1) - 1):
                stage_probs[:, i] = cum_probs[:, i - 1] - cum_probs[:, i]
            stage_probs[:, -1] = cum_probs[:, -1]
            stage_probs = stage_probs.clamp_min(0.0)
            stage_probs = stage_probs / (stage_probs.sum(dim=1, keepdim=True) + 1e-8)
            if class_idx is None:
                class_idx = stage_probs.argmax(dim=1)[0].item()
            score = stage_probs[:, int(class_idx)].sum()
        elif logits.shape[-1] == 1 or (len(logits.shape) == 2 and logits.shape[1] == 1):
            if use_sigmoid:
                score = torch.sigmoid(logits).squeeze(1).sum()
            else:
                score = logits.squeeze(1).sum()
        else:
            if class_idx is None:
                class_idx = logits.argmax(dim=1)[0].item()
            score = logits[:, int(class_idx)].sum()

        self.model.zero_grad()
        score.backward(retain_graph=False)

        feat_map = self.model._feat_maps.get(self.target_layer_name)
        gradient = self.model._gradients.get(self.target_layer_name)

        if feat_map is None or gradient is None:
            raise RuntimeError(f"未找到目标层 '{self.target_layer_name}'")

        grad_sq = gradient ** 2
        grad_cu = gradient ** 3

        spatial_sum = feat_map.sum(dim=[2, 3, 4], keepdim=True).clamp(min=1e-8)
        alpha = grad_sq / (2 * grad_sq + spatial_sum * grad_cu + 1e-8)

        positive_gradient = F.relu(gradient)
        weights = (alpha * positive_gradient).sum(dim=[2, 3, 4], keepdim=True)

        cam = (weights * feat_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=xl_t1.shape[2:], mode='trilinear', align_corners=False)

        cam = cam.squeeze(1).detach().cpu().numpy()
        for i in range(cam.shape[0]):
            p_low = np.percentile(cam[i], 1)
            p_high = np.percentile(cam[i], 99)
            if p_high > p_low:
                cam[i] = np.clip((cam[i] - p_low) / (p_high - p_low), 0, 1)
            else:
                vmin, vmax = cam[i].min(), cam[i].max()
                if vmax > vmin:
                    cam[i] = (cam[i] - vmin) / (vmax - vmin)

        return cam
class DiffMapVisualizer:

    @staticmethod
    def get_diff_heatmap(diff_abs: torch.Tensor) -> np.ndarray:
        heatmap = diff_abs.mean(dim=1, keepdim=True)
        heatmap = heatmap.detach().cpu().float()

        heatmap_np = heatmap.squeeze(1).numpy()
        for i in range(heatmap_np.shape[0]):
            vmin, vmax = heatmap_np[i].min(), heatmap_np[i].max()
            if vmax > vmin:
                heatmap_np[i] = (heatmap_np[i] - vmin) / (vmax - vmin)

        return heatmap_np


def save_cam_slices(
    cam_3d: np.ndarray,
    save_path: str,
    image_3d: Optional[np.ndarray] = None,
    n_slices: int = 5,
    colormap: str = 'jet',
    alpha_overlay: float = 0.4,
    threshold: Optional[float] = None,
    title: str = 'GradCAM',
    dpi: int = 180,
) -> None:

    D, H, W = cam_3d.shape
    cmap = plt.get_cmap(colormap)

    def get_indices(length: int, n: int):
        if length <= 1:
            return [0]
        return np.round(np.linspace(length*0.2, length*0.8, n)).astype(int).clip(0, length-1)

    slices_d = get_indices(D, n_slices)
    slices_h = get_indices(H, n_slices)
    slices_w = get_indices(W, n_slices)

    fig, axes = plt.subplots(3, n_slices,
                             figsize=(n_slices * 2.8, 9),
                             constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', fontfamily='Times New Roman', y=1.02)

    row_labels = ['Axial (D)', 'Coronal (H)', 'Sagittal (W)']
    slice_getters = [
        lambda idx: cam_3d[idx, :, :],
        lambda idx: cam_3d[:, idx, :],
        lambda idx: cam_3d[:, :, idx]
    ]
    img_getters = [
        lambda idx: image_3d[idx, :, :] if image_3d is not None else None,
        lambda idx: image_3d[:, idx, :] if image_3d is not None else None,
        lambda idx: image_3d[:, :, idx] if image_3d is not None else None
    ]

    for row, (label, slice_indices, get_cam, get_img) in enumerate(zip(
            row_labels, [slices_d, slices_h, slices_w], slice_getters, img_getters)):

        for col, idx in enumerate(slice_indices):
            ax = axes[row, col]
            cam_slice = get_cam(idx)

            if image_3d is not None:
                img_slice = get_img(idx)
                img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
                ax.imshow(img_norm, cmap='gray', aspect='auto')

                cam_rgba = cmap(np.clip(cam_slice, 0, 1))
                if threshold is not None:
                    cam_rgba[..., 3] = np.where(cam_slice >= threshold, alpha_overlay, 0)
                else:
                    cam_rgba[..., 3] = alpha_overlay * (0.35 + 0.65 * np.clip(cam_slice, 0, 1))

                ax.imshow(cam_rgba, aspect='auto')
            else:
                ax.imshow(cam_slice, cmap=colormap, vmin=0, vmax=1, aspect='auto')

            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold', fontfamily='Times New Roman')
            ax.set_title(f'idx={idx}', fontsize=8, fontweight='bold', fontfamily='Times New Roman')
            ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.65, aspect=25, pad=0.03)
    cbar.set_label('Activation Intensity', fontsize=9, fontweight='bold', fontfamily='Times New Roman')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[CAM] Saved: {save_path}")



def save_diff_map_slices(
    diff_heatmap: np.ndarray,
    save_path: str,
    n_slices: int = 5,
    colormap: str = 'hot',
    title: str = 'Difference Map |ΔF|',
    dpi: int = 180,
) -> None:
    save_cam_slices(
        cam_3d=diff_heatmap,
        save_path=save_path,
        image_3d=None,
        n_slices=n_slices,
        colormap=colormap,
        alpha_overlay=1.0,
        title=title,
        dpi=dpi,
    )


def compare_bilateral_cam(
        cam_left: np.ndarray,
        cam_right: np.ndarray,
        save_path: str,
        image_left: Optional[np.ndarray] = None,
        image_right: Optional[np.ndarray] = None,
        axis: int = 1,
        n_slices: int = 5,
        alpha_overlay: float = 0.45,
        title: str = 'Bilateral CAM Comparison',
        dpi: int = 300,
) -> None:
    shape = cam_left.shape
    slice_indices = np.linspace(int(shape[axis] * 0.2), int(shape[axis] * 0.8),
                                n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices, figsize=(n_slices * 3, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold', fontfamily='Times New Roman')

    cmap = cm.get_cmap('jet')

    for col, idx in enumerate(slice_indices):
        sl = [slice(None)] * 3
        sl[axis] = idx
        sl = tuple(sl)

        if image_left is not None:
            img_slice = image_left[sl]
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            axes[0, col].imshow(img_norm, cmap='gray', aspect='auto')
            cam_rgba = cmap(cam_left[sl])
            cam_rgba[..., 3] = alpha_overlay * (0.35 + 0.65 * np.clip(cam_left[sl], 0, 1))
            axes[0, col].imshow(cam_rgba, aspect='auto')
        else:
            axes[0, col].imshow(cam_left[sl], cmap='jet', vmin=0, vmax=1, aspect='auto')

        axes[0, col].set_title(f'Left idx={idx}', fontsize=8, fontweight='bold', fontfamily='Times New Roman')
        axes[0, col].axis('off')

        if image_right is not None:
            img_slice = image_right[sl]
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            axes[1, col].imshow(img_norm, cmap='gray', aspect='auto')
            cam_rgba = cmap(cam_right[sl])
            cam_rgba[..., 3] = alpha_overlay * (0.35 + 0.65 * np.clip(cam_right[sl], 0, 1))
            axes[1, col].imshow(cam_rgba, aspect='auto')
        else:
            axes[1, col].imshow(cam_right[sl], cmap='jet', vmin=0, vmax=1, aspect='auto')

        axes[1, col].set_title(f'Right idx={idx}', fontsize=8, fontweight='bold', fontfamily='Times New Roman')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Left Hip', fontsize=9, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 0].set_ylabel('Right Hip', fontsize=9, fontweight='bold', fontfamily='Times New Roman')

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, aspect=20, pad=0.02)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontweight('bold')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[CAM] Bilateral comparison saved: {save_path}")
