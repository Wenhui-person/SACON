
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AblationConfig:

    use_contralateral: bool = True
    registration_type: str = "deformable"
    use_gated_fusion: bool = True
    use_explicit_difference: bool = True
    use_diff_attention: bool = True
    use_structure_signal_decoupling: bool = True
    use_local_diff_gate: bool = True
    use_concentration_loss: bool = False
    capacity_control: str = "none"

    def __post_init__(self) -> None:
        if self.registration_type not in {"none", "flip", "deformable"}:
            raise ValueError(f"未知配准类型: {self.registration_type}")
        if not self.use_contralateral:
            self.registration_type = "none"
            self.use_explicit_difference = False
            self.use_diff_attention = False
            self.use_structure_signal_decoupling = False
            self.use_local_diff_gate = False
            self.use_concentration_loss = False
        if not self.use_explicit_difference:
            self.use_diff_attention = False
            self.use_local_diff_gate = False


@dataclass
class TrainConfig:

    data_root: str = r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_2026126_Aug\onfh_data.json"
    base_channels: int = 32
    num_stages: int = 3
    class_names: Tuple[str, str, str] = ("Normal", "Early", "Late")
    batch_size: int = 2
    num_epochs: int = 100
    lr: float = 8.96e-5
    weight_decay: float = 1e-4
    volume_size: Tuple[int, int, int] = (19, 314, 224)
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    save_dir: str = "./checkpoints"
    experiment_root: Optional[str] = None
    selected_folds: Optional[Tuple[int, ...]] = None
    random_seed: int = 42
    device: str = "cuda"
    ssl_t1_ckpt: Optional[str] = None
    ssl_t2_ckpt: Optional[str] = None
    freeze_encoder_epochs: int = 0
    run_tag: str = ""

    alpha_bin: float = 1.2
    alpha_stage: float = 0.82
    alpha_reg: float = 0.18
    alpha_decouple: float = 0.07
    alpha_conc: float = 0.0
    alpha_local_conc: float = 0.0
    ordinal: bool = True

    augment: bool = True
    online_train_augment: bool = False
    late_aug_copies: int = 5
    nonlate_aug_copies: int = 1
    late_aug_max_retries: int = 30
    figure_dir: str = "./outputs/figures"
    patience: int = 10
    every_save_time: int = 5
    feature_cloud_interval: Optional[int] = 0
    feature_cloud_3d: bool = False
    save_best_feature_cloud: bool = False
    grad_flow_interval: Optional[int] = 0
    grad_flow_samples: int = 1
    tensorboard_image_interval: int = 0
    progress_update_interval: int = 20
    enable_cam_hooks: bool = False
    plot_interval: int = 0
    plot_on_best: bool = True
    task: str = "stage"

    group_by_base_id: bool = False
    repeated_holdout_7_3: bool = False
    real_only_test: bool = True
    real_test_ratio: float = 0.3


def get_ablation_configs() -> Dict[str, AblationConfig]:

    common = dict(
        use_contralateral=True,
        registration_type="deformable",
        use_gated_fusion=True,
        use_explicit_difference=True,
        use_diff_attention=True,
        use_structure_signal_decoupling=True,
        use_local_diff_gate=True,
        use_concentration_loss=False,
    )

    def variant(**changes) -> AblationConfig:
        return AblationConfig(**{**common, **changes})

    return {
        "full": variant(),
        "no_contralateral": AblationConfig(use_contralateral=False),
        "bilateral_no_diff": variant(use_explicit_difference=False),
        "no_registration": variant(registration_type="none"),
        "flip_only": variant(registration_type="flip"),
        "no_modal_gate": variant(use_gated_fusion=False),
        "no_diff_attention": variant(use_diff_attention=False),
        "no_decoupling": variant(use_structure_signal_decoupling=False),
        "no_local_diff_gate": variant(use_local_diff_gate=False),
        "bilateral_capacity_matched": variant(capacity_control="bilateral"),
        "sacon_small": variant(capacity_control="small"),
    }
