
import math
import os
from contextlib import AbstractContextManager
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_FEATURE_METHODS = ("mean", "mid")

DEFAULT_OUTPUT_FEATURES = (
    "left_t1_layer3",
    "right_t1_layer3",
    "left_t2_layer3",
    "right_t2_layer3",
    "f_left",
    "f_right",
    "f_left_str",
    "f_right_str",
    "f_left_sig",
    "f_right_sig",
    "diff_feat",
    "diff_aware",
    "diff_abs",
    "z_left",
    "z_right",
    "z_left_local",
    "z_right_local",
    "left_diff_gate",
    "right_diff_gate",
)


class FeatureMapHookCollector(AbstractContextManager):

    def __init__(self, model):
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._t1_call_idx = 0
        self._t2_call_idx = 0

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
        return False

    def register(self) -> None:
        self.remove()
        self.features.clear()
        self._t1_call_idx = 0
        self._t2_call_idx = 0

        if hasattr(self.model, "t1_encoder") and hasattr(self.model.t1_encoder, "layer3_conv"):
            self._hooks.append(
                self.model.t1_encoder.layer3_conv.register_forward_hook(
                    self._make_encoder_hook("t1")
                )
            )
        if hasattr(self.model, "t2_encoder") and hasattr(self.model.t2_encoder, "layer3_conv"):
            self._hooks.append(
                self.model.t2_encoder.layer3_conv.register_forward_hook(
                    self._make_encoder_hook("t2")
                )
            )

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_encoder_hook(self, modality: str):
        names = {
            "t1": ("left_t1_layer3", "right_t1_layer3"),
            "t2": ("left_t2_layer3", "right_t2_layer3"),
        }[modality]

        def hook(_module, _inputs, output):
            if modality == "t1":
                call_idx = self._t1_call_idx
                self._t1_call_idx += 1
            else:
                call_idx = self._t2_call_idx
                self._t2_call_idx += 1

            if call_idx < len(names):
                self.features[names[call_idx]] = output.detach().cpu()
            else:
                self.features[f"{modality}_layer3_call{call_idx}"] = output.detach().cpu()

        return hook


def normalize_feature_map(feature_2d: torch.Tensor) -> torch.Tensor:
    x = feature_2d.detach().float().cpu()
    x = x - x.min()
    x_max = x.max()
    if float(x_max) > 0:
        x = x / (x_max + 1e-8)
    return x


def project_feature_volume(feature: torch.Tensor, method: str = "mean") -> torch.Tensor:
    if method not in {"mean", "max", "mid"}:
        raise ValueError("method must be one of: mean, max, mid")

    if feature.dim() == 3:
        if method == "mean":
            return feature.mean(dim=0)
        if method == "max":
            return feature.max(dim=0).values
        return feature[feature.shape[0] // 2]

    if feature.dim() == 4:
        if method == "mean":
            return feature.mean(dim=1)
        if method == "max":
            return feature.max(dim=1).values
        return feature[:, feature.shape[1] // 2]

    raise ValueError(f"Expected [C,D,H,W] or [D,H,W], got shape {tuple(feature.shape)}")


def make_channel_grid(
    feature: torch.Tensor,
    method: str = "mean",
    max_channels: int = 16,
    n_cols: int = 4,
    upsample_to: Optional[Sequence[int]] = None,
    channel_padding: Optional[int] = None,
) -> np.ndarray:
    if feature.dim() == 5:
        feature = feature[0]
    if feature.dim() == 3:
        feature = feature.unsqueeze(0)
    if feature.dim() != 4:
        raise ValueError(f"Expected [B,C,D,H,W], [C,D,H,W], or [D,H,W], got {tuple(feature.shape)}")

    feature = feature.detach().float().cpu()[:max_channels]
    if upsample_to is not None:
        feature = F.interpolate(
            feature.unsqueeze(0),
            size=tuple(upsample_to),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

    projected = project_feature_volume(feature, method=method)
    channels = [normalize_feature_map(ch).numpy() for ch in projected]

    if not channels:
        raise ValueError("No channels available for visualization.")

    height, width = channels[0].shape
    n_cols = max(1, min(n_cols, len(channels)))
    n_rows = math.ceil(len(channels) / n_cols)

    if channel_padding is None:
        padding = max(8, int(round(min(height, width) * 0.04)))
    else:
        padding = max(0, int(channel_padding))

    grid = np.full(
        (
            n_rows * height + (n_rows - 1) * padding,
            n_cols * width + (n_cols - 1) * padding,
        ),
        np.nan,
        dtype=np.float32,
    )

    for idx, channel in enumerate(channels):
        row = idx // n_cols
        col = idx % n_cols
        y0 = row * (height + padding)
        x0 = col * (width + padding)
        grid[y0 : y0 + height, x0 : x0 + width] = channel

    return grid


def save_feature_map_grid(
    feature: torch.Tensor,
    save_path: str,
    feature_name: str,
    method: str = "mean",
    max_channels: int = 16,
    n_cols: int = 4,
    upsample_to: Optional[Sequence[int]] = None,
    channel_padding: Optional[int] = None,
    dpi: int = 180,
) -> str:
    grid = make_channel_grid(
        feature=feature,
        method=method,
        max_channels=max_channels,
        n_cols=n_cols,
        upsample_to=upsample_to,
        channel_padding=channel_padding,
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    plt.figure(figsize=(max(6, grid.shape[1] / 60), max(4, grid.shape[0] / 60)))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.axis("off")
    plt.title(f"{feature_name} | {method}", fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    return save_path


def _safe_feature_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _collect_output_features(outputs: Mapping[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collected = {}
    for name in DEFAULT_OUTPUT_FEATURES:
        value = outputs.get(name)
        if isinstance(value, torch.Tensor) and value.dim() in (3, 4, 5):
            collected[name] = value.detach().cpu()
    return collected


def save_sacon_feature_maps(
    outputs: Mapping[str, Optional[torch.Tensor]],
    save_dir: str,
    encoder_features: Optional[Mapping[str, torch.Tensor]] = None,
    methods: Iterable[str] = DEFAULT_FEATURE_METHODS,
    max_channels: int = 16,
    selected_features: Optional[Iterable[str]] = None,
    n_cols: int = 4,
    reference_size: Optional[Sequence[int]] = None,
    channel_padding: Optional[int] = None,
) -> Dict[str, List[str]]:
    all_features = {}
    if encoder_features:
        all_features.update(
            {
                name: feat.detach().cpu()
                for name, feat in encoder_features.items()
                if isinstance(feat, torch.Tensor) and feat.dim() in (3, 4, 5)
            }
        )
    all_features.update(_collect_output_features(outputs))

    if selected_features is None:
        selected = list(DEFAULT_OUTPUT_FEATURES)
    else:
        selected = list(selected_features)

    for encoder_name in (
        "left_t1_layer3",
        "right_t1_layer3",
        "left_t2_layer3",
        "right_t2_layer3",
    ):
        if encoder_name not in selected:
            selected.insert(0, encoder_name)

    os.makedirs(save_dir, exist_ok=True)
    saved: Dict[str, List[str]] = {}
    for name in selected:
        feature = all_features.get(name)
        if feature is None:
            continue

        saved[name] = []
        for method in methods:
            method = method.strip().lower()
            if not method:
                continue
            save_path = os.path.join(save_dir, f"{_safe_feature_filename(name)}_{method}.png")
            try:
                saved[name].append(
                    save_feature_map_grid(
                        feature=feature,
                        save_path=save_path,
                        feature_name=name,
                        method=method,
                        max_channels=max_channels,
                        n_cols=n_cols,
                        upsample_to=reference_size,
                        channel_padding=channel_padding,
                    )
                )
            except ValueError as exc:
                print(f"[FeatureMap] Skip {name} ({method}): {exc}")

        if not saved[name]:
            saved.pop(name, None)

    print(f"[FeatureMap] Saved {sum(len(paths) for paths in saved.values())} PNG files to: {save_dir}")
    return saved
