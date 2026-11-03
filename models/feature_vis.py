
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class SACONFeatureVisualizer:

    def __init__(self, writer: SummaryWriter, max_channels: int = 8):
        self.writer = writer
        self.max_channels = max_channels

    @staticmethod
    def normalize_map(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x - x.min()
        if x.max() > 0:
            x = x / (x.max() + 1e-8)
        return x

    def extract_mid_slice(self, vol: torch.Tensor) -> torch.Tensor:
        if vol.dim() == 3:
            d = vol.shape[0] // 2
            return vol[d]
        elif vol.dim() == 4:
            d = vol.shape[1] // 2
            return vol[:, d]
        else:
            raise ValueError(f"Unsupported shape: {vol.shape}")

    def project_depth(self, vol: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if mode not in ["mean", "max"]:
            raise ValueError("mode must be 'mean' or 'max'")

        if vol.dim() == 3:
            if mode == "mean":
                return vol.mean(dim=0)
            else:
                return vol.max(dim=0).values

        elif vol.dim() == 4:
            if mode == "mean":
                return vol.mean(dim=1)
            else:
                return vol.max(dim=1).values

        else:
            raise ValueError(f"Unsupported shape: {vol.shape}")

    def tensor3d_to_grid(
        self,
        feat: torch.Tensor,
        method: str = "mean",
        max_channels: int = None,
        nrow: int = 4
    ) -> torch.Tensor:
        assert feat.dim() == 4, f"Expected [C, D, H, W], got {feat.shape}"

        if max_channels is None:
            max_channels = self.max_channels

        feat = feat[:max_channels]

        if method == "mean":
            feat_2d = self.project_depth(feat, mode="mean")
        elif method == "max":
            feat_2d = self.project_depth(feat, mode="max")
        elif method == "mid":
            feat_2d = self.extract_mid_slice(feat)
        else:
            raise ValueError("method must be one of ['mean', 'max', 'mid']")

        feat_2d = torch.stack([self.normalize_map(ch) for ch in feat_2d], dim=0)

        feat_2d = feat_2d.unsqueeze(1)

        grid = make_grid(feat_2d, nrow=nrow, padding=2)
        return grid

    def add_feature_map(
        self,
        tag: str,
        feat: torch.Tensor,
        global_step: int,
        method: str = "mean",
        max_channels: int = None
    ):
        if feat is None:
            return
        assert feat.dim() == 5, f"{tag} should be [B, C, D, H, W], got {feat.shape}"

        sample = feat[0].detach().cpu()
        grid = self.tensor3d_to_grid(sample, method=method, max_channels=max_channels)
        self.writer.add_image(tag, grid, global_step)

    def add_volume_slice(
        self,
        tag: str,
        vol: torch.Tensor,
        global_step: int
    ):
        if vol is None:
            return
        assert vol.dim() == 5 and vol.shape[1] == 1, f"{tag} should be [B,1,D,H,W], got {vol.shape}"

        sample = vol[0, 0].detach().cpu()
        img = self.extract_mid_slice(sample)
        img = self.normalize_map(img)
        self.writer.add_image(tag, img.unsqueeze(0), global_step)

    def add_displacement_field(
        self,
        tag: str,
        disp: torch.Tensor,
        global_step: int
    ):
        if disp is None:
            return
        assert disp.dim() == 5 and disp.shape[1] == 3, f"{tag} should be [B,3,D,H,W], got {disp.shape}"

        sample = disp[0].detach().cpu()
        imgs = self.extract_mid_slice(sample)

        imgs = torch.stack([self.normalize_map(ch) for ch in imgs], dim=0)
        self.writer.add_image(tag, imgs, global_step)

    def compute_gradcam_3d(
        self,
        feat_map: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        cam = (weights * feat_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min = cam.amin(dim=(2, 3, 4), keepdim=True)
        cam_max = cam.amax(dim=(2, 3, 4), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def add_gradcam(
            self,
            tag: str,
            cam: torch.Tensor,
            image: torch.Tensor,
            global_step: int
    ):
        assert cam.dim() == 5 and cam.shape[1] == 1
        assert image.dim() == 5 and image.shape[1] == 1

        cam_up = F.interpolate(
            cam.detach(),
            size=image.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        cam_img = self.extract_mid_slice(cam_up[0, 0].cpu())
        src_img = self.extract_mid_slice(image[0, 0].detach().cpu())

        cam_img = self.normalize_map(cam_img)
        src_img = self.normalize_map(src_img)

        cam_rgb = torch.stack([cam_img, torch.zeros_like(cam_img), torch.zeros_like(cam_img)], dim=0)
        src_rgb = src_img.unsqueeze(0).repeat(3, 1, 1)


        overlay = torch.stack([
            cam_img,
            0.5 * src_img + 0.5 * cam_img,
            src_img
        ], dim=0)

        self.writer.add_image(f"{tag}/image", src_rgb, global_step)
        self.writer.add_image(f"{tag}/cam", cam_rgb, global_step)
        self.writer.add_image(f"{tag}/overlay", overlay, global_step)


def visualize_sacon_step(
    model,
    outputs: dict,
    xl_t1: torch.Tensor,
    xl_t2: torch.Tensor,
    xr_t1: torch.Tensor,
    xr_t2: torch.Tensor,
    writer: SummaryWriter,
    global_step: int
):

    visualizer = SACONFeatureVisualizer(writer, max_channels=8)

    visualizer.add_volume_slice("input/left_t1", xl_t1, global_step)
    visualizer.add_volume_slice("input/left_t2", xl_t2, global_step)
    visualizer.add_volume_slice("input/right_t1", torch.flip(xr_t1,dims=[-1]), global_step)
    visualizer.add_volume_slice("input/right_t2", torch.flip(xr_t2,dims=[-1]), global_step)

    if "xl_t1_aligned" in outputs:
        visualizer.add_volume_slice("aligned/left_t1", outputs["xl_t1_aligned"], global_step)
    if "xl_t2_aligned" in outputs:
        visualizer.add_volume_slice("aligned/left_t2", outputs["xl_t2_aligned"], global_step)
    if "xr_t1_aligned" in outputs:
        visualizer.add_volume_slice("aligned/right_t1", outputs["xr_t1_aligned"], global_step)
    if "xr_t2_aligned" in outputs:
        visualizer.add_volume_slice("aligned/right_t2", outputs["xr_t2_aligned"], global_step)

    if "disp_fields" in outputs and outputs["disp_fields"] is not None:
        visualizer.add_displacement_field("alignment/disp_fields", outputs["disp_fields"], global_step)


    if hasattr(model, "_feat_maps"):
        if "t1_layer3" in model._feat_maps:
            visualizer.add_feature_map(
                "encoder/t1_layer3_mean",
                model._feat_maps["t1_layer3"],
                global_step,
                method="mean",
                max_channels=8
            )
            visualizer.add_feature_map(
                "encoder/t1_layer3_mid",
                model._feat_maps["t1_layer3"],
                global_step,
                method="mid",
                max_channels=8
            )

        if "t2_layer3" in model._feat_maps:
            visualizer.add_feature_map(
                "encoder/t2_layer3_mean",
                model._feat_maps["t2_layer3"],
                global_step,
                method="mean",
                max_channels=8
            )
            visualizer.add_feature_map(
                "encoder/t2_layer3_mid",
                model._feat_maps["t2_layer3"],
                global_step,
                method="mid",
                max_channels=8
            )

    visualizer.add_feature_map("fusion/f_left", outputs["f_left"], global_step, method="mean")
    visualizer.add_feature_map("fusion/f_right", outputs["f_right"], global_step, method="mean")

    visualizer.add_feature_map("decoupler/f_left_str", outputs["f_left_str"], global_step, method="mean")
    visualizer.add_feature_map("decoupler/f_right_str", outputs["f_right_str"], global_step, method="mean")
    visualizer.add_feature_map("decoupler/f_left_sig", outputs["f_left_sig"], global_step, method="mean")
    visualizer.add_feature_map("decoupler/f_right_sig", outputs["f_right_sig"], global_step, method="mean")

    visualizer.add_feature_map("contra/diff_feat", outputs["diff_feat"], global_step, method="mean")
    visualizer.add_feature_map("contra/diff_aware", outputs["diff_aware"], global_step, method="mean")
    visualizer.add_feature_map("contra/diff_abs", outputs["diff_abs"], global_step, method="mean")

    visualizer.add_feature_map("head/z_left", outputs["z_left"], global_step, method="mean")
    visualizer.add_feature_map("head/z_right", outputs["z_right"], global_step, method="mean")


def log_sacon_gradcam(
    model,
    outputs: dict,
    xl_t1: torch.Tensor,
    xl_t2: torch.Tensor,
    xr_t1: torch.Tensor,
    xr_t2: torch.Tensor,
    writer: SummaryWriter,
    global_step: int
):
    visualizer = SACONFeatureVisualizer(writer, max_channels=16)

    if "t1_layer3" in model._feat_maps and "t1_layer3" in model._gradients:
        cam_t1 = visualizer.compute_gradcam_3d(
            model._feat_maps["t1_layer3"].detach(),
            model._gradients["t1_layer3"].detach()
        )
        visualizer.add_gradcam("gradcam/left_t1", cam_t1, xl_t1, global_step)
        visualizer.add_gradcam("gradcam/right_t1", cam_t1, xl_t2, global_step)

    if "t2_layer3" in model._feat_maps and "t2_layer3" in model._gradients:
        cam_t2 = visualizer.compute_gradcam_3d(
            model._feat_maps["t2_layer3"].detach(),
            model._gradients["t2_layer3"].detach()
        )
        visualizer.add_gradcam("gradcam/left_t2", cam_t2, xl_t2, global_step)
        visualizer.add_gradcam("gradcam/right_t2", cam_t2, xl_t2, global_step)


GRADIENT_FLOW_KEYS = ("f_left", "f_right", "diff_feat", "diff_aware", "z_left", "z_right")


def retain_gradient_flow(outputs: dict) -> None:
    for key in GRADIENT_FLOW_KEYS:
        feat = outputs.get(key)
        if isinstance(feat, torch.Tensor) and feat.requires_grad:
            feat.retain_grad()


def _grad_norm(feat: torch.Tensor) -> float:
    if not isinstance(feat, torch.Tensor) or feat.grad is None:
        return 0.0
    return float(feat.grad.detach().float().norm().cpu())


def collect_gradient_flow_metrics(outputs: dict, epoch: int, step: int) -> dict:
    grad_left = _grad_norm(outputs.get("f_left"))
    grad_right = _grad_norm(outputs.get("f_right"))
    grad_diff = _grad_norm(outputs.get("diff_feat"))
    grad_diff_aware = _grad_norm(outputs.get("diff_aware"))
    grad_z_left = _grad_norm(outputs.get("z_left"))
    grad_z_right = _grad_norm(outputs.get("z_right"))

    eps = 1e-8
    return {
        "epoch": epoch,
        "step": step,
        "grad_norm_left": grad_left,
        "grad_norm_right": grad_right,
        "grad_norm_diff": grad_diff,
        "grad_norm_diff_aware": grad_diff_aware,
        "grad_norm_z_left": grad_z_left,
        "grad_norm_z_right": grad_z_right,
        "left_right_grad_ratio": grad_left / (grad_right + eps),
        "interaction_share": grad_diff_aware / (grad_left + grad_right + grad_diff_aware + eps),
        "gradient_balance": 1.0 - abs(grad_left - grad_right) / (grad_left + grad_right + eps),
    }


def append_gradient_flow_metrics(row: dict, csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        csv_path,
        mode="a",
        index=False,
        header=not os.path.exists(csv_path),
    )


def plot_gradient_flow_curves(csv_path: str, save_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(df))
    axes[0].plot(x, df["grad_norm_left"], label="Left", lw=1.8)
    axes[0].plot(x, df["grad_norm_right"], label="Right", lw=1.8)
    axes[0].set_title("Branch Grad Norm", fontweight="bold")
    axes[0].set_ylabel("L2 norm")
    axes[0].legend(frameon=False)

    axes[1].plot(x, df["grad_norm_diff"], label="diff_feat", lw=1.8)
    axes[1].plot(x, df["grad_norm_diff_aware"], label="diff_aware", lw=1.8)
    axes[1].set_title("Interaction Path", fontweight="bold")
    axes[1].legend(frameon=False)

    axes[2].plot(x, df["interaction_share"], label="Interaction share", lw=1.8)
    axes[2].plot(x, df["gradient_balance"], label="Balance", lw=1.8)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Interaction Diagnostics", fontweight="bold")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("Logged step")
        ax.grid(alpha=0.25)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _grad_heatmap(feat: torch.Tensor, sample_idx: int = 0):
    if not isinstance(feat, torch.Tensor) or feat.grad is None or feat.grad.dim() != 5:
        return None
    grad = feat.grad.detach().float()[sample_idx].abs().mean(dim=0).cpu()
    if grad.numel() == 0:
        return None
    d = grad.shape[0] // 2
    img = grad[d].numpy()
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin + 1e-8)
    return img


def _image_slice(image: torch.Tensor, sample_idx: int = 0):
    if not isinstance(image, torch.Tensor) or image.dim() != 5:
        return None
    img = image.detach().float()[sample_idx].cpu()
    img = img[0] if img.shape[0] == 1 else img.mean(dim=0)
    d = img.shape[0] // 2
    img = img[d].numpy()
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin + 1e-8)
    return img


def _resolve_image(images, name: str):
    if isinstance(images, dict):
        return images.get(name)
    if name == "left_t1":
        return images
    return None


def save_gradient_flow_figure(
        outputs: dict,
        save_path: str,
        max_samples: int = 1,
        images: torch.Tensor = None,
        dpi: int = 600,
        annotate_sides: bool = False,
        sample_annotations=None,
) -> None:
    keys = ("f_left", "f_right", "diff_feat", "diff_aware", "z_left", "z_right")
    batch_size = 1
    for feat in outputs.values():
        if isinstance(feat, torch.Tensor) and feat.dim() == 5:
            batch_size = feat.shape[0]
            break
    sample_count = max(1, min(max_samples, batch_size))

    image_columns = ("Left T1", "Left T2", "Left Fusion", "Right T1", "Right T2", "Right Fusion")
    columns = image_columns + keys
    fig, axes = plt.subplots(
        sample_count * 2,
        len(columns),
        figsize=(len(columns) * 1.22, sample_count * 2.35),
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)
    for row in range(sample_count):
        top_row = row * 2
        bottom_row = top_row + 1
        left_t1 = _resolve_image(images, "left_t1")
        left_t2 = _resolve_image(images, "left_t2")
        right_t1 = _resolve_image(images, "right_t1")
        right_t2 = _resolve_image(images, "right_t2")
        left_fusion = _resolve_image(images, "left_fusion")
        right_fusion = _resolve_image(images, "right_fusion")
        image_map = {
            "Left T1": left_t1,
            "Left T2": left_t2,
            "Left Fusion": left_fusion,
            "Right T1": right_t1,
            "Right T2": right_t2,
            "Right Fusion": right_fusion,
        }

        for col, title in enumerate(image_columns):
            ax = axes[top_row, col]
            src_img = _image_slice(image_map.get(title), row)
            if src_img is not None:
                if title in {"Right T1", "Right T2", "Right Fusion"}:
                    src_img = np.fliplr(src_img)
                ax.imshow(src_img, cmap="gray", vmin=0, vmax=1)
                if annotate_sides and title in {"Left T1", "Left T2", "Right T1", "Right T2"}:
                    side_label = "L" if title.startswith("Left") else "R"
                    side_text = ax.text(
                        0.04,
                        0.05,
                        side_label,
                        transform=ax.transAxes,
                        color="white",
                        fontfamily="Arial",
                        fontsize=8,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                    )
                    side_text.set_path_effects(
                        [path_effects.Stroke(linewidth=1.2, foreground="black"), path_effects.Normal()]
                    )
            ax.set_title(title, fontsize=6, fontweight="bold", pad=1)
            ax.axis("off")
            axes[bottom_row, col].axis("off")

        for col, key in enumerate(keys, start=len(image_columns)):
            axes[top_row, col].axis("off")
            ax = axes[bottom_row, col]
            img = _grad_heatmap(outputs.get(key), row)
            if img is not None:
                if key in {"f_right", "z_right"}:
                    img = np.fliplr(img)
                ax.imshow(img, cmap="magma", vmin=0, vmax=1)
            ax.set_title(key, fontsize=6, fontweight="bold", pad=1)
            ax.axis("off")

        if sample_annotations and row < len(sample_annotations):
            axes[bottom_row, 0].text(
                0.0,
                0.72,
                str(sample_annotations[row]),
                transform=axes[bottom_row, 0].transAxes,
                fontfamily="Arial",
                fontsize=6.5,
                color="black",
                ha="left",
                va="top",
                linespacing=1.35,
                clip_on=False,
            )

    fig.suptitle("Gradient Flow Between Bilateral Branches", fontsize=9, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.003, right=0.955, bottom=0.004, top=0.91, wspace=0.012, hspace=0.012)
    sm = plt.cm.ScalarMappable(cmap="magma")
    sm.set_clim(0, 1)
    cax = fig.add_axes([0.965, 0.08, 0.012, 0.75])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Gradient", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=max(601, int(dpi)), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
