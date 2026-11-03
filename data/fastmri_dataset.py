
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import h5py
except ImportError as exc:
    raise ImportError("Please install h5py before using fastMRI: pip install h5py") from exc


DEFAULT_SPLITS = ("multicoil_train", "singlecoil_train")
RECON_KEYS = ("reconstruction_rss", "reconstruction_esc")


def parse_splits(value: str) -> Tuple[str, ...]:
    if value is None or value.strip() == "":
        return DEFAULT_SPLITS
    splits = tuple(split.strip() for split in value.split(",") if split.strip())
    return splits or DEFAULT_SPLITS


def _decode_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_value(value.item())
        return " ".join(_decode_value(v) for v in value.tolist())
    return str(value)


def _select_recon_key(h5_file: h5py.File) -> str:
    for key in RECON_KEYS:
        if key in h5_file:
            return key
    raise KeyError(f"No image-space reconstruction found. Expected one of {RECON_KEYS}.")


def _metadata_from_h5(path: str, split: str) -> Optional[Dict[str, str]]:
    try:
        with h5py.File(path, "r") as h5_file:
            recon_key = _select_recon_key(h5_file)
            shape = h5_file[recon_key].shape
            if len(shape) < 3:
                raise ValueError(f"Expected 3D reconstruction, got shape={shape}")
            attrs = {str(k).lower(): _decode_value(v) for k, v in h5_file.attrs.items()}
            return {
                "path": path,
                "split": split,
                "acquisition": attrs.get("acquisition", ""),
                "patient_id": attrs.get("patient_id", ""),
                "recon_key": recon_key,
                "num_slices": str(shape[-3]),
                "height": str(shape[-2]),
                "width": str(shape[-1]),
            }
    except Exception as exc:
        print(f"[fastMRIDataset] Skipped unreadable H5: {path} | {exc}")
        return None


def discover_fastmri_h5_files(
    fastmri_root: str,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> List[Dict[str, str]]:
    rows = []
    for split in splits:
        split_root = os.path.join(fastmri_root, split)
        if not os.path.isdir(split_root):
            print(f"[fastMRIDataset] Split directory not found, skipping: {split_root}")
            continue
        for root, _, files in os.walk(split_root):
            for name in files:
                if not name.lower().endswith((".h5", ".hdf5")):
                    continue
                path = os.path.join(root, name)
                row = _metadata_from_h5(path, split)
                if row is not None:
                    rows.append(row)
    return sorted(rows, key=lambda row: (row["split"], row["path"]))


def build_fastmri_manifest(
    fastmri_root: str,
    manifest_path: str,
    splits: Sequence[str] = DEFAULT_SPLITS,
    max_files: Optional[int] = None,
) -> List[Dict[str, str]]:
    rows = discover_fastmri_h5_files(fastmri_root, splits=splits)
    if max_files is not None and max_files > 0:
        rows = rows[:max_files]

    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "split",
                "acquisition",
                "patient_id",
                "recon_key",
                "num_slices",
                "height",
                "width",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_manifest(manifest_path: str) -> List[Dict[str, str]]:
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if np.iscomplexobj(volume):
        volume = np.abs(volume)
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if volume.ndim == 2:
        volume = volume[None, ...]
    while volume.ndim > 3:
        volume = np.squeeze(volume)
        if volume.ndim <= 3:
            break
        volume = np.sqrt(np.sum(np.square(volume), axis=0))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 2D/3D reconstructed image volume, got shape {volume.shape}")

    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    v_min, v_max = float(volume.min()), float(volume.max())
    if v_max - v_min > 1e-6:
        volume = (volume - v_min) / (v_max - v_min)
    return volume.astype(np.float32)


def _resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)
    return tensor.squeeze(0)


class FastMRIDataset(Dataset):

    def __init__(
        self,
        fastmri_root: Optional[str] = None,
        manifest_path: Optional[str] = None,
        splits: Sequence[str] = DEFAULT_SPLITS,
        volume_size: Tuple[int, int, int] = (19, 314, 224),
        max_files: Optional[int] = None,
    ):
        if manifest_path:
            rows = _read_manifest(manifest_path)
        elif fastmri_root:
            rows = discover_fastmri_h5_files(fastmri_root, splits=splits)
        else:
            raise ValueError("Either fastmri_root or manifest_path must be provided.")

        allowed_splits = {split.strip() for split in splits if split.strip()}
        if allowed_splits:
            rows = [row for row in rows if row.get("split", "") in allowed_splits]

        if max_files is not None and max_files > 0:
            rows = rows[:max_files]

        self.rows = rows
        self.volume_size = volume_size
        if not self.rows:
            raise ValueError("No fastMRI H5 files found for the requested split filter.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.rows[idx]
        path = row["path"]
        recon_key = row.get("recon_key") or "reconstruction_rss"
        with h5py.File(path, "r") as h5_file:
            if recon_key not in h5_file:
                recon_key = _select_recon_key(h5_file)
            volume = h5_file[recon_key][()]

        volume = _normalize_volume(volume)
        return _resize_volume(volume, self.volume_size)
