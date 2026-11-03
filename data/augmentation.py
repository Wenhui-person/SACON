
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Optional, Sequence, Tuple, Union


MIN_MAE_TO_ORIGINAL = 0.003
MAX_CORR_TO_ORIGINAL = 0.9995
MIN_PAIRWISE_MAE = 0.001
MAX_PAIRWISE_CORR = 0.9998
DEFAULT_MAX_RETRIES = 30
DEFAULT_LATE_LABEL = 2
DEFAULT_LATE_AUG_COPIES = 5
DEFAULT_NONLATE_AUG_COPIES = 1
DEFAULT_IMAGE_KEYS = ("left_t1", "left_t2", "right_t1", "right_t2")


ArrayLikeImages = Union[Sequence[np.ndarray], Dict[str, np.ndarray]]


def is_late_case(left_els: int, right_els: int, late_label: int = DEFAULT_LATE_LABEL) -> bool:
    return int(left_els) == late_label or int(right_els) == late_label


def build_online_augmentation_index(
    source_indices: Sequence[int],
    left_els_labels: Sequence[int],
    right_els_labels: Sequence[int],
    late_aug_copies: int = DEFAULT_LATE_AUG_COPIES,
    nonlate_aug_copies: int = DEFAULT_NONLATE_AUG_COPIES,
    late_label: int = DEFAULT_LATE_LABEL,
) -> List[Tuple[int, int]]:
    virtual_index = []
    for source_idx in source_indices:
        left_label = left_els_labels[source_idx]
        right_label = right_els_labels[source_idx]
        copies = late_aug_copies if is_late_case(left_label, right_label, late_label) else nonlate_aug_copies
        virtual_index.append((int(source_idx), 0))
        for copy_id in range(1, int(copies) + 1):
            virtual_index.append((int(source_idx), copy_id))
    return virtual_index


def _image_keys_for(images: ArrayLikeImages) -> Tuple[str, ...]:
    if isinstance(images, dict):
        return tuple(images.keys())
    return DEFAULT_IMAGE_KEYS[:len(images)]


def _as_array_dict(images: ArrayLikeImages) -> Dict[str, np.ndarray]:
    if isinstance(images, dict):
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in images.items()}
    keys = _image_keys_for(images)
    return {key: np.asarray(value, dtype=np.float32) for key, value in zip(keys, images)}


def _restore_image_container(reference: ArrayLikeImages, arrays: Dict[str, np.ndarray]):
    if isinstance(reference, dict):
        return {key: arrays[key] for key in reference.keys()}
    keys = _image_keys_for(reference)
    return tuple(arrays[key] for key in keys)


def corrcoef_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare_arrays(reference: ArrayLikeImages, candidate: ArrayLikeImages) -> List[Dict[str, float]]:
    ref_arrays = _as_array_dict(reference)
    cand_arrays = _as_array_dict(candidate)
    rows = []
    for key in ref_arrays:
        ref = ref_arrays[key]
        cand = cand_arrays[key]
        diff = cand - ref
        rows.append({
            "modality": key,
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "corr": corrcoef_flat(ref, cand),
        })
    return rows


def aggregate_comparison(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    maes = np.asarray([row["mae"] for row in rows], dtype=np.float32)
    rmses = np.asarray([row["rmse"] for row in rows], dtype=np.float32)
    corrs = np.asarray([row["corr"] for row in rows], dtype=np.float32)
    return {
        "mean_mae": float(maes.mean()),
        "mean_rmse": float(rmses.mean()),
        "mean_corr": float(corrs.mean()),
        "max_corr": float(corrs.max()),
    }


def candidate_differs_from_original(
    original_arrays: ArrayLikeImages,
    candidate_arrays: ArrayLikeImages,
    min_mae: float = MIN_MAE_TO_ORIGINAL,
    max_corr: float = MAX_CORR_TO_ORIGINAL,
) -> Tuple[bool, Dict[str, float], List[Dict[str, float]]]:
    comparison = compare_arrays(original_arrays, candidate_arrays)
    metrics = aggregate_comparison(comparison)
    accepted = metrics["mean_mae"] >= min_mae and metrics["max_corr"] <= max_corr
    return accepted, metrics, comparison


def candidate_differs_from_previous(
    previous_arrays_list: Sequence[ArrayLikeImages],
    candidate_arrays: ArrayLikeImages,
    min_pairwise_mae: float = MIN_PAIRWISE_MAE,
    max_pairwise_corr: float = MAX_PAIRWISE_CORR,
) -> Tuple[bool, Dict[str, float]]:
    if not previous_arrays_list:
        return True, {"min_pairwise_mae": np.nan, "max_pairwise_corr": np.nan}

    pair_maes = []
    pair_corrs = []
    for previous_arrays in previous_arrays_list:
        comparison = compare_arrays(previous_arrays, candidate_arrays)
        pair_maes.append(float(np.mean([row["mae"] for row in comparison])))
        pair_corrs.append(float(np.max([row["corr"] for row in comparison])))

    metrics = {
        "min_pairwise_mae": float(np.min(pair_maes)),
        "max_pairwise_corr": float(np.max(pair_corrs)),
    }
    accepted = (
        metrics["min_pairwise_mae"] >= min_pairwise_mae
        and metrics["max_pairwise_corr"] <= max_pairwise_corr
    )
    return accepted, metrics


class OfflineStyleMRITrainAugmentation:

    def __init__(
        self,
        bias_coefficients: float = 0.5,
        log_gamma: Tuple[float, float] = (0.3, 0.5),
        noise_std: Tuple[float, float] = (0.01, 0.05),
        blur_std: Tuple[float, float] = (0.5, 1.8),
        min_mae_to_original: float = MIN_MAE_TO_ORIGINAL,
        max_corr_to_original: float = MAX_CORR_TO_ORIGINAL,
        min_pairwise_mae: float = MIN_PAIRWISE_MAE,
        max_pairwise_corr: float = MAX_PAIRWISE_CORR,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        try:
            import torchio as tio
        except ImportError as exc:
            raise ImportError(
                "OfflineStyleMRITrainAugmentation requires torchio. "
                "Install it with `pip install torchio` or install requirements.txt."
            ) from exc

        self.tio = tio
        self.bias_coefficients = bias_coefficients
        self.log_gamma = log_gamma
        self.noise_std = noise_std
        self.blur_std = blur_std
        self.min_mae_to_original = min_mae_to_original
        self.max_corr_to_original = max_corr_to_original
        self.min_pairwise_mae = min_pairwise_mae
        self.max_pairwise_corr = max_pairwise_corr
        self.max_retries = int(max_retries)

    def make_transform(self):
        return self.tio.Compose([
            self.tio.RandomBiasField(coefficients=self.bias_coefficients, p=1.0),
            self.tio.RandomGamma(log_gamma=self.log_gamma, p=1.0),
            self.tio.RandomNoise(std=self.noise_std, p=1.0),
            self.tio.RandomBlur(std=self.blur_std, p=1.0),
        ])

    def _to_subject(self, images: ArrayLikeImages):
        arrays = _as_array_dict(images)
        subject = {}
        for key, array in arrays.items():
            if array.ndim == 3:
                tensor = torch.from_numpy(array[None].astype(np.float32))
            elif array.ndim == 4:
                tensor = torch.from_numpy(array.astype(np.float32))
            else:
                raise ValueError(f"Expected 3D or 4D volume for {key}, got shape {array.shape}.")
            subject[key] = self.tio.ScalarImage(tensor=tensor)
        return self.tio.Subject(subject)

    def _from_subject(self, subject, reference: ArrayLikeImages) -> Dict[str, np.ndarray]:
        reference_arrays = _as_array_dict(reference)
        output = {}
        for key, ref in reference_arrays.items():
            array = subject[key].data.detach().cpu().numpy().astype(np.float32)
            if ref.ndim == 3 and array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            output[key] = array
        return output

    def _augment_once(self, images: ArrayLikeImages, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is None:
            subject = self.make_transform()(self._to_subject(images))
            return self._from_subject(subject, images)

        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            np.random.seed(seed)
            torch.manual_seed(seed)
            subject = self.make_transform()(self._to_subject(images))
            return self._from_subject(subject, images)
        finally:
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

    def generate_copies(
        self,
        images: ArrayLikeImages,
        num_copies: int,
        base_seed: Optional[int] = None,
        return_qc: bool = False,
    ):
        original = _as_array_dict(images)
        accepted_arrays = []
        qc_rows = []

        for copy_id in range(1, int(num_copies) + 1):
            accepted = False
            last_qc = None
            for retry in range(self.max_retries):
                seed = None if base_seed is None else int(base_seed) + copy_id * 100 + retry
                candidate = self._augment_once(original, seed=seed)

                ok_original, original_metrics, _ = candidate_differs_from_original(
                    original,
                    candidate,
                    min_mae=self.min_mae_to_original,
                    max_corr=self.max_corr_to_original,
                )
                ok_pairwise, pairwise_metrics = candidate_differs_from_previous(
                    accepted_arrays,
                    candidate,
                    min_pairwise_mae=self.min_pairwise_mae,
                    max_pairwise_corr=self.max_pairwise_corr,
                )
                last_qc = {
                    "copy_id": copy_id,
                    "retry": retry,
                    "seed": seed,
                    "accepted": ok_original and ok_pairwise,
                    **original_metrics,
                    **pairwise_metrics,
                }
                if ok_original and ok_pairwise:
                    accepted_arrays.append(candidate)
                    qc_rows.append(last_qc)
                    accepted = True
                    break

            if not accepted:
                raise RuntimeError(
                    f"Failed to generate distinct augmentation copy {copy_id} "
                    f"after {self.max_retries} retries. Last QC: {last_qc}"
                )

        copies = [_restore_image_container(images, arrays) for arrays in accepted_arrays]
        if return_qc:
            return copies, qc_rows
        return copies

    def __call__(
        self,
        images: ArrayLikeImages,
        copy_id: int = 1,
        previous_arrays_list: Optional[Sequence[ArrayLikeImages]] = None,
        seed: Optional[int] = None,
    ):
        if int(copy_id) == 0:
            arrays = _as_array_dict(images)
            return _restore_image_container(images, {key: value.copy() for key, value in arrays.items()})

        previous_arrays_list = previous_arrays_list or []
        original = _as_array_dict(images)
        for retry in range(self.max_retries):
            retry_seed = None if seed is None else int(seed) + retry
            candidate = self._augment_once(original, seed=retry_seed)
            ok_original, _, _ = candidate_differs_from_original(
                original,
                candidate,
                min_mae=self.min_mae_to_original,
                max_corr=self.max_corr_to_original,
            )
            ok_pairwise, _ = candidate_differs_from_previous(
                previous_arrays_list,
                candidate,
                min_pairwise_mae=self.min_pairwise_mae,
                max_pairwise_corr=self.max_pairwise_corr,
            )
            if ok_original and ok_pairwise:
                return _restore_image_container(images, candidate)
        raise RuntimeError(f"Failed to generate distinct augmentation copy_id={copy_id}.")


class MRIAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images):


        if np.random.rand() < self.prob:
            images = [self.intensity_augment(image) for image in images]
        if np.random.rand() < self.prob:
            images = [self.bias_field(image) for image in images]
        if np.random.rand() < self.prob:
            images = [self.add_noise(image) for image in images]
        if np.random.rand() < self.prob:
            images = [self.blur(image) for image in images]
        if np.random.rand() < self.prob:
            images = [self.motion_artifact(image) for image in images]
        return images[0],images[1],images[2],images[3]

    def intensity_augment(self, image):
        gamma = np.random.uniform(0.8, 1.2)
        image = np.power(image, gamma)

        contrast = np.random.uniform(0.9, 1.1)
        brightness = np.random.uniform(-0.1, 0.1)
        image = image * contrast + brightness

        return np.clip(image, 0, 1)

    def bias_field(self, image):
        shape = image.shape[-3:]

        field = np.random.randn(*[s // 4 for s in shape])
        field = gaussian_filter(field, sigma=2)

        from scipy.ndimage import zoom
        zoom_factors = [s / f for s, f in zip(shape, field.shape)]
        field = zoom(field, zoom_factors, order=1)

        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        field = 0.8 + 0.4 * field

        if image.ndim == 4:
            field = field[None, ...]

        return image * field

    def add_noise(self, image):
        sigma = np.random.uniform(0.01, 0.05)

        noise_real = np.random.normal(0, sigma, image.shape)
        noise_imag = np.random.normal(0, sigma, image.shape)
        noisy = np.sqrt((image + noise_real) ** 2 + noise_imag ** 2)

        return noisy

    def blur(self, image):
        sigma = np.random.uniform(0.2, 1.0)

        if image.ndim == 4:
            for c in range(image.shape[0]):
                image[c] = gaussian_filter(image[c], sigma=sigma)
        else:
            image = gaussian_filter(image, sigma=sigma)

        return image

    def motion_artifact(self, image):
        if image.ndim == 4:
            img = image[0]
        else:
            img = image

        axis = np.random.choice([0, 1, 2])
        shift = np.random.randint(5, 15)
        intensity = np.random.uniform(0.1, 0.3)

        ghost = np.roll(img, shift, axis=axis)
        img = img + intensity * ghost

        if image.ndim == 4:
            image[0] = img
        else:
            image = img

        return np.clip(image, 0, 1)


if __name__ == '__main__':

    augment = MRIAugmentation(prob=0.5)

    image = torch.rand(1, 64, 128, 128)
    augmented = augment(image)
