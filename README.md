# SACON

Symmetry-Aware Contralateral-difference Analysis for osteonecrosis of the femoral head using bilateral T1/T2 MRI. This PyTorch implementation supports binary classification and disease staging through symmetry alignment, multimodal fusion, and contralateral feature interaction.

## Installation

Use Python 3.10 or later and a PyTorch installation compatible with your hardware.

```bash
python -m pip install -r requirements.txt
python -m pip install SimpleITK scipy seaborn tqdm tensorboard
```

Optional packages: `optuna` for hyperparameter optimization, `umap-learn` for UMAP, `openpyxl` for Excel input, and `python-docx` / `pydicom` for the corresponding reporting scripts. Word automation requires Windows, Microsoft Word, and `pywin32`.

## Data

Prepare preprocessed left/right T1 and T2 NIfTI volumes and a JSON manifest containing parallel lists of equal length:

```json
{
  "patient_id": ["subject_001"],
  "left_t1_path": ["/path/to/subject_001/left_t1.nii.gz"],
  "left_t2_path": ["/path/to/subject_001/left_t2.nii.gz"],
  "right_t1_path": ["/path/to/subject_001/right_t1.nii.gz"],
  "right_t2_path": ["/path/to/subject_001/right_t2.nii.gz"],
  "label_left_bin": [0],
  "label_right_bin": [1],
  "label_left_ELS": [0],
  "label_right_ELS": [1]
}
```

Binary labels are `0` (normal) and `1` (ONFH). Default staging labels are `0` (Normal), `1` (Early), and `2` (Late). 

## Training

```bash
python train.py --data_root /path/to/manifest.json --task stage --ablation full --epochs 100 --batch_size 2 --save_dir ./checkpoints --ssl_ckpt= --run_tag= --feature_cloud_interval 0
```

Use `--task bin` for binary classification. This example disables external SSL initialization and feature-cloud visualization. To initialize from SSL weights, set `--ssl_ckpt /path/to/encoder.pth`, or supply `--ssl_t1_ckpt` and `--ssl_t2_ckpt` separately.

Training and ablation configurations are in `configs/config.py`. Run `python train.py --help` for split, augmentation, and logging options. Set split options explicitly to match your experimental protocol. Training saves checkpoints, split records, and metrics under the selected output directory.

## Inference

Run staging inference and CAM visualization using a trained checkpoint and its saved fold split:

```bash
python inference.py --ckpt /path/to/stage_full_best.pth --data_root /path/to/manifest.json --fold 0 --split_csv /path/to/fold_data_split.csv --out_dir ./outputs/inference
```

Keep manifest ordering consistent with training. `evaluate.py` currently uses `DummyDataset` and is a demonstration entry point. Subgroup evaluation is available through `evaluate_subgroups.py`; inspect its subcommand help for required data and demographic inputs. Some scripts retain local default paths, so pass paths explicitly for your environment.

## Repository structure

| Path | Purpose |
| --- | --- |
| `models/` | SACON architecture and feature visualization |
| `data/` | Dataset loaders, augmentation, and preprocessing notebooks |
| `configs/` | Training and ablation configurations |
| `losses/` | Training objectives |
| `utils/` | Metrics, plotting, SSL loading, and subgroup analysis |
| `scripts/` | Reporting and verification tools |
| `pretrain_fastmri*_ssl.py` | Self-supervised pretraining |
| `uih_supervised_adaptation.py` | Supervised adaptation experiments |
| `ablation_study.py` | Ablation experiments |
