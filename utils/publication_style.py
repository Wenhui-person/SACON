
from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

MM_TO_INCH = 1.0 / 25.4
SINGLE_COLUMN_MM = 89
DOUBLE_COLUMN_MM = 183
MAX_HEIGHT_MM = 220

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}
STAGE_COLORS = [COLORS["gray"], COLORS["blue"], COLORS["vermillion"]]
STRATEGY_COLORS = {
    "zero_shot": COLORS["gray"],
    "siemens_finetune": COLORS["blue"],
    "uih_ssl_only": COLORS["orange"],
}
STRATEGY_MARKERS = {"zero_shot": "o", "siemens_finetune": "s", "uih_ssl_only": "^"}
STRATEGY_LINESTYLES = {"zero_shot": "--", "siemens_finetune": "-", "uih_ssl_only": "-."}


def apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def create_nature_figure(width: str = "double", height_mm: float = 90, **kwargs):
    width_mm = SINGLE_COLUMN_MM if width == "single" else DOUBLE_COLUMN_MM
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"图高不能超过 {MAX_HEIGHT_MM} mm")
    apply_nature_style()
    return plt.subplots(figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH), **kwargs)


def clean_axis(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6, zorder=0)


def add_panel_labels(axes: Iterable, labels: Optional[Iterable[str]] = None) -> None:
    axes = list(axes)
    labels = list(labels or [chr(97 + i) for i in range(len(axes))])
    for ax, label in zip(axes, labels):
        ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")


def save_publication_figure(fig, output_stem, raster_dpi: int = 600) -> list[str]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".pdf", ".svg", ".png"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=raster_dpi if suffix == ".png" else None,
                    bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    return paths


def save_source_data(frame, output_path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate source-data columns: {duplicates}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def audit_figure(path) -> list[str]:
    path = Path(path)
    issues = []
    if path.suffix.lower() not in {".pdf", ".svg", ".png"}:
        issues.append("不支持的正式图格式")
    if not path.exists() or path.stat().st_size == 0:
        issues.append("文件不存在或为空")
    if path.suffix.lower() == ".png" and path.exists():
        try:
            from PIL import Image
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))
                if min(dpi) < 300:
                    issues.append(f"PNG 分辨率低于 300 dpi: {dpi}")
                if image.mode not in {"RGB", "RGBA"}:
                    issues.append(f"PNG 不是 RGB/RGBA: {image.mode}")
        except ImportError:
            issues.append("未安装 Pillow，无法自动核验 PNG dpi/色彩模式")
    return issues
