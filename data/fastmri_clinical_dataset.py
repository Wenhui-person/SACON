
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import pydicom
except ImportError as exc:
    raise ImportError("Please install pydicom before using clinical fastMRI DICOM data.") from exc


DEFAULT_CLINICAL_ROOTS = (
    r"G:\公共医学影像数据\fastMRI\knee_mri_clinical_seq",
    r"G:\公共医学影像数据\fastMRI\knee_mri_clinical_seq_batch2",
)

MANIFEST_FIELDS = [
    "series_dir",
    "source_batch",
    "patient_dir",
    "study_id",
    "series_id",
    "series_description",
    "plane",
    "contrast_group",
    "num_slices",
    "height",
    "width",
]


def parse_clinical_roots(value: str) -> Tuple[str, ...]:
    if value is None or value.strip() == "":
        return DEFAULT_CLINICAL_ROOTS
    text = value.strip()
    delimiter = os.pathsep if os.pathsep in text else ","
    roots = tuple(part.strip().strip('"') for part in text.split(delimiter) if part.strip())
    return roots or DEFAULT_CLINICAL_ROOTS


def parse_filter(value: str) -> Tuple[str, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def infer_plane(series_description: str) -> str:
    desc = (series_description or "").upper().replace("_", " ")
    if "AX" in desc or "AXIAL" in desc:
        return "AX"
    if "COR" in desc or "CORONAL" in desc:
        return "COR"
    if "SAG" in desc or "SAGITTAL" in desc:
        return "SAG"
    return ""


def infer_contrast_group(series_description: str) -> str:
    desc = (series_description or "").upper().replace("_", " ")
    has_fs = " FS" in f" {desc} " or "FATSAT" in desc or "FAT SAT" in desc
    if "T1" in desc:
        return "T1_FS" if has_fs else "T1"
    if "T2" in desc:
        return "T2_FS" if has_fs else "T2"
    if "PD" in desc:
        return "PD_FS" if has_fs else "PD"
    return ""


def _first_dicom_file(series_dir: str) -> Optional[str]:
    try:
        for name in sorted(os.listdir(series_dir)):
            if name.lower().endswith(".dcm"):
                return os.path.join(series_dir, name)
    except OSError:
        return None
    return None


def _safe_get_str(ds, name: str) -> str:
    value = getattr(ds, name, "")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value) if value is not None else ""


def _metadata_from_series(series_dir: str, source_root: str) -> Optional[Dict[str, str]]:
    first_file = _first_dicom_file(series_dir)
    if not first_file:
        return None

    try:
        ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
        series_description = _safe_get_str(ds, "SeriesDescription")
        rows = _safe_get_str(ds, "Rows")
        cols = _safe_get_str(ds, "Columns")
    except Exception as exc:
        print(f"[FastMRIClinical] Skipped unreadable DICOM series: {series_dir} | {exc}")
        return None

    try:
        num_slices = sum(1 for name in os.listdir(series_dir) if name.lower().endswith(".dcm"))
    except OSError as exc:
        print(f"[FastMRIClinical] Skipped inaccessible DICOM series: {series_dir} | {exc}")
        return None

    if num_slices <= 0:
        return None

    parent = os.path.dirname(series_dir)
    patient_dir = os.path.basename(os.path.dirname(parent))
    return {
        "series_dir": series_dir,
        "source_batch": os.path.basename(os.path.normpath(source_root)),
        "patient_dir": patient_dir,
        "study_id": os.path.basename(parent),
        "series_id": os.path.basename(series_dir),
        "series_description": series_description,
        "plane": infer_plane(series_description),
        "contrast_group": infer_contrast_group(series_description),
        "num_slices": str(num_slices),
        "height": rows,
        "width": cols,
    }


def discover_fastmri_clinical_series(
    clinical_roots: Sequence[str] = DEFAULT_CLINICAL_ROOTS,
    max_series: Optional[int] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for source_root in clinical_roots:
        if not os.path.isdir(source_root):
            print(f"[FastMRIClinical] Root not found, skipping: {source_root}")
            continue
        root_name = os.path.basename(os.path.normpath(source_root)) or source_root
        with tqdm(desc=f"Scanning {root_name}", unit="dirs", dynamic_ncols=True) as pbar:
            for root, dirs, _ in os.walk(source_root):
                pbar.update(1)
                dirs.sort()
                if not os.path.basename(root).startswith("MR"):
                    continue
                row = _metadata_from_series(root, source_root)
                if row is not None:
                    rows.append(row)
                    pbar.set_postfix({"series": len(rows)})
                    if max_series is not None and max_series > 0 and len(rows) >= max_series:
                        return rows
    return rows


def build_fastmri_clinical_manifest(
    clinical_roots: Sequence[str],
    manifest_path: str,
    max_series: Optional[int] = None,
) -> List[Dict[str, str]]:
    rows = discover_fastmri_clinical_series(clinical_roots, max_series=max_series)
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_manifest(manifest_path: str) -> List[Dict[str, str]]:
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_dicom_volume(series_dir: str) -> np.ndarray:
    files = [
        os.path.join(series_dir, name)
        for name in os.listdir(series_dir)
        if name.lower().endswith(".dcm")
    ]
    if not files:
        raise ValueError(f"No DICOM files found in series: {series_dir}")

    slices = []
    for path in files:
        ds = pydicom.dcmread(path, force=True)
        image = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        image = image * slope + intercept
        instance = int(getattr(ds, "InstanceNumber", 0) or 0)
        location = float(getattr(ds, "SliceLocation", instance) or instance)
        slices.append((instance, location, path, image))

    slices.sort(key=lambda item: (item[0], item[1], item[2]))
    shapes = {item[3].shape for item in slices}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent slice shapes in {series_dir}: {sorted(shapes)}")
    return np.stack([item[3] for item in slices], axis=0)


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    v_min, v_max = float(volume.min()), float(volume.max())
    if v_max - v_min > 1e-6:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
    return volume.astype(np.float32)


def _resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)
    return tensor.squeeze(0)


class FastMRIClinicalDataset(Dataset):

    def __init__(
        self,
        clinical_roots: Sequence[str] = DEFAULT_CLINICAL_ROOTS,
        manifest_path: Optional[str] = None,
        volume_size: Tuple[int, int, int] = (19, 314, 224),
        max_series: Optional[int] = None,
        include_planes: Sequence[str] = (),
        include_contrasts: Sequence[str] = (),
    ):
        if manifest_path:
            rows = _read_manifest(manifest_path)
        else:
            rows = discover_fastmri_clinical_series(clinical_roots, max_series=max_series)

        allowed_planes = {plane.upper() for plane in include_planes if plane}
        allowed_contrasts = {contrast.upper() for contrast in include_contrasts if contrast}
        if allowed_planes:
            rows = [row for row in rows if row.get("plane", "").upper() in allowed_planes]
        if allowed_contrasts:
            rows = [row for row in rows if row.get("contrast_group", "").upper() in allowed_contrasts]
        if max_series is not None and max_series > 0:
            rows = rows[:max_series]

        if not rows:
            raise ValueError("No fastMRI clinical DICOM series found for the requested filters.")
        self.rows = rows
        self.volume_size = volume_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.rows[idx]
        volume = _read_dicom_volume(row["series_dir"])
        volume = _normalize_volume(volume)
        return _resize_volume(volume, self.volume_size)
