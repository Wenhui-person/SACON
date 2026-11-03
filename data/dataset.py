
import json
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, OrderedDict, List
import SimpleITK as sitk
from .augmentation import (
    DEFAULT_LATE_AUG_COPIES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_NONLATE_AUG_COPIES,
    MRIAugmentation,
    OfflineStyleMRITrainAugmentation,
    build_online_augmentation_index,
)

class ONFHDataset(Dataset):

    CLASS_MAP = {
        "Normal": 0,
        "Early": 1,
        "Late": 2,
    }

    def __init__(
        self,
        data_root: str,
        ids: List,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        augment: bool = True,
        online_train_augment: bool = False,
        late_aug_copies: int = DEFAULT_LATE_AUG_COPIES,
        nonlate_aug_copies: int = DEFAULT_NONLATE_AUG_COPIES,
        late_aug_max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("请安装 SimpleITK: pip install SimpleITK")

        self.data_root = data_root
        self.ids = ids
        self.volume_size = volume_size
        self.online_train_augment = online_train_augment
        self.augment = augment and not online_train_augment
        self.augmentation = MRIAugmentation(prob=0.5)
        self.online_augmentation = None
        self.online_aug_cache = {}

        self.samples = self._load_sample_list()
        if self.online_train_augment:
            self.virtual_index = build_online_augmentation_index(
                source_indices=list(self.ids),
                left_els_labels=self.samples["label_left_ELS"],
                right_els_labels=self.samples["label_right_ELS"],
                late_aug_copies=late_aug_copies,
                nonlate_aug_copies=nonlate_aug_copies,
            )
            self.online_augmentation = OfflineStyleMRITrainAugmentation(
                max_retries=late_aug_max_retries,
            )
            aug_count = sum(1 for _, copy_id in self.virtual_index if copy_id > 0)
            print(
                f"[ONFHDataset] online_train_augment=True | "
                f"source_samples={len(self.ids)} virtual_samples={len(self.virtual_index)} "
                f"augmented_copies={aug_count}"
            )
        else:
            self.virtual_index = [(int(idx), 0) for idx in self.ids]

        self.original_left_stage_class_counts, self.original_right_stage_class_counts = \
            self._compute_class_counts(self.ids)
        self.left_stage_class_counts,self.right_stage_class_counts = self._compute_class_counts()
        self.left_bin_class_counts,self.right_bin_class_counts = self._compute_neg_pos_counts()

    def _count_source_indices(self):
        if self.online_train_augment:
            return [source_idx for source_idx, _ in self.virtual_index]
        return self.ids

    def _compute_class_counts(self, source_indices=None):
        left_counts = {}
        right_counts = {}
        count_indices = self._count_source_indices() if source_indices is None else source_indices
        for idx in count_indices:
            label = self.samples["label_left_ELS"][idx]
            left_counts[label] = left_counts.get(label, 0) + 1
        for idx in count_indices:
            label = self.samples["label_right_ELS"][idx]
            right_counts[label] = right_counts.get(label, 0) + 1
        return dict(sorted(left_counts.items())),dict(sorted(right_counts.items()))

    def _compute_neg_pos_counts(self, source_indices=None):
        left_counts = {}
        right_counts = {}
        count_indices = self._count_source_indices() if source_indices is None else source_indices
        for idx in count_indices:
            label = self.samples["label_left_bin"][idx]
            left_counts[label] = left_counts.get(label, 0) + 1
        for idx in count_indices:
            label = self.samples["label_right_bin"][idx]
            right_counts[label] = right_counts.get(label, 0) + 1
        return dict(sorted(left_counts.items())),dict(sorted(right_counts.items()))

    def _load_sample_list(self):
        assert os.path.exists(self.data_root), f"Warning: {self.data_root} not found. Creating dummy data for demo."

        with open(self.data_root, 'r') as f:
            samples = json.load(f,object_pairs_hook=OrderedDict)


        return samples

    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)

        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min > 1e-6:
            volume = (volume - v_min) / (v_max - v_min)


        return volume

    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        target_d, target_h, target_w = self.volume_size
        src_d, src_h, src_w = volume.shape

        zoom_factors = (
            target_d / src_d,
            target_h / src_h,
            target_w / src_w
        )

        volume = ndimage.zoom(volume, zoom_factors, order=1)
        return volume

    def get_class_weights(self,class_counts) -> torch.Tensor:
        num_classes = len(self.CLASS_MAP)
        counts = np.array([class_counts.get(i, 1) for i in range(num_classes)])
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes
        return torch.FloatTensor(weights)

    def get_bin_class_weights(self,class_counts) -> torch.Tensor:
        counts = np.array([class_counts.get(i, 1) for i in range(2)])
        weights = 1.0 / counts
        weights = weights / weights.sum() * 2
        return torch.FloatTensor(weights)

    def return_stage_class_weights(self):
        return (
            self.get_class_weights(self.original_left_stage_class_counts),
            self.get_class_weights(self.original_right_stage_class_counts),
        )
    def return_bin_class_weights(self):
        return self.get_bin_class_weights(self.left_bin_class_counts),self.get_bin_class_weights(self.right_bin_class_counts)

    def __len__(self) -> int:
        return len(self.virtual_index) if self.online_train_augment else len(self.ids)

    def _load_volume(self, path: str) -> np.ndarray:
        import torch.nn.functional as F
        image = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(image)
        resized_data = self.resize_volume(data)
        predata = self._preprocess(resized_data)

        return predata

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.online_train_augment:
            idx, copy_id = self.virtual_index[idx]
        else:
            idx = self.ids[idx]
            copy_id = 0

        left_t1  = self._load_volume(self.samples['left_t1_path'][idx])
        left_t2  = self._load_volume(self.samples['left_t2_path'][idx])
        right_t1 = self._load_volume(self.samples['right_t1_path'][idx])
        right_t2 = self._load_volume(self.samples['right_t2_path'][idx])

        if self.online_train_augment and copy_id > 0:
            if idx not in self.online_aug_cache:
                max_copy_id = max(
                    virtual_copy_id
                    for source_idx, virtual_copy_id in self.virtual_index
                    if source_idx == idx
                )
                copies = self.online_augmentation.generate_copies(
                    [left_t1, left_t2, right_t1, right_t2],
                    num_copies=max_copy_id,
                    base_seed=int(idx) * 10000,
                )
                self.online_aug_cache[idx] = {
                    generated_copy_id: generated_images
                    for generated_copy_id, generated_images in enumerate(copies, start=1)
                }
            left_t1, left_t2, right_t1, right_t2 = self.online_aug_cache[idx][copy_id]
        elif self.augment:
            left_t1,left_t2,right_t1,right_t2 = self.augmentation([left_t1,left_t2,right_t1,right_t2])

        left_t1 = torch.from_numpy(left_t1.astype(np.float32)).unsqueeze(0)
        left_t2 = torch.from_numpy(left_t2.astype(np.float32)).unsqueeze(0)
        right_t1 = torch.from_numpy(right_t1.astype(np.float32)).unsqueeze(0)
        right_t2 = torch.from_numpy(right_t2.astype(np.float32)).unsqueeze(0)

        return {
            'left_t1':      left_t1,
            'left_t2':      left_t2,
            'right_t1':     right_t1,
            'right_t2':     right_t2,
            'left_binary':  torch.tensor(self.samples['label_left_bin'][idx], dtype=torch.float32),
            'left_stage':   torch.tensor(self.samples['label_left_ELS'][idx], dtype=torch.long),
            'right_binary': torch.tensor(self.samples['label_right_bin'][idx], dtype=torch.float32),
            'right_stage':  torch.tensor(self.samples['label_right_ELS'][idx], dtype=torch.long),
            'label_left_ELS': torch.tensor(self.samples['label_left_ELS'][idx], dtype=torch.long),
            'label_right_ELS': torch.tensor(self.samples['label_right_ELS'][idx], dtype=torch.long),
            'patient_id': str(self.samples['patient_id'][idx]),
            'source_index': torch.tensor(idx, dtype=torch.long),
            'copy_id': torch.tensor(copy_id, dtype=torch.long),
            'is_augmented': torch.tensor(copy_id > 0, dtype=torch.bool),
        }


class DummyDataset(Dataset):

    def __init__(
        self,
        num_samples: int = 100,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        num_stages: int = 5,
    ):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.num_stages = num_stages

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'left_t1':      torch.randn(1, *self.volume_size),
            'left_t2':      torch.randn(1, *self.volume_size),
            'right_t1':     torch.randn(1, *self.volume_size),
            'right_t2':     torch.randn(1, *self.volume_size),
            'left_binary':  torch.randint(0, 2, ()).float(),
            'left_stage':   torch.randint(0, self.num_stages, ()),
            'right_binary': torch.randint(0, 2, ()).float(),
            'right_stage':  torch.randint(0, self.num_stages, ()),
            'subject_id':   str(idx),
        }
