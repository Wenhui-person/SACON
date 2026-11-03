import logging

import optuna
import os
import sys
import argparse
import json
from typing import Dict, Tuple, OrderedDict, List

import torch
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from models.feature_vis import visualize_sacon_step, log_sacon_gradcam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AblationConfig, TrainConfig, get_ablation_configs
from models.sacon import SACON
from losses.losses import SACANLoss
from data.dataset import ONFHDataset
from utils.metrics import compute_binary_metrics, compute_stage_metrics, MetricTracker, plot_confusion_matrix, \
    plot_roc_curves, plot_precision_recall_curves, plot_binary_classification_metrics, plot_class_distribution, \
    ordinal_logits_to_class_probs, dca_curve, plot_dca, calibration_metrics, plot_calibration
from utils.dimensionality_reduction import visualize_pca_with_density, visualize_umap, visualize_tsne
from sklearn.metrics import roc_auc_score
import numpy as np


def train_one_epoch(
        model: SACON,
        loader: data_utils.DataLoader,
        criterion: SACANLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        total_steps: int,
        writer,
        epoch: int,
        vis_interval=10,
        ordinal=False
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    loss_sums = {'binary': 0., 'stage': 0., 'registration': 0., 'concentration': 0.}

    scaler = GradScaler(enabled=True)

    loop = tqdm(loader, total=len(loader))

    for batch in loop:
        total_steps += 1
        xl_t1 = batch['left_t1'].to(device)
        xl_t2 = batch['left_t2'].to(device)
        xr_t1 = batch['right_t1'].to(device)
        xr_t2 = batch['right_t2'].to(device)


        labels = {
            'left_binary': batch['left_binary'].to(device),
            'left_stage': batch['label_left_ELS'].to(device),
            'right_binary': batch['right_binary'].to(device),
            'right_stage': batch['label_right_ELS'].to(device),
        }

        optimizer.zero_grad()
        with autocast(enabled=True):
            outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)

            loss, loss_dict = criterion(outputs, labels)
            if ordinal:
                outputs["left_stage"] = ordinal_logits_to_class_probs(outputs["left_stage"], num_classes=3)
                outputs["right_stage"] = ordinal_logits_to_class_probs(outputs["right_stage"], num_classes=3)


        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_sums:
            component = loss_dict.get(k, 0.)
            loss_sums[k] += (
                float(component.detach().item())
                if isinstance(component, torch.Tensor)
                else component
            )

        loop.set_postfix({'total_loss': loss.item(), 'binary': loss_dict['binary'], 'stage': loss_dict['stage'],
                          'registration': loss_dict['registration'], 'concentration': loss_dict['concentration']})


    n = len(loader)

    return {'total': total_loss / n, **{k: v / n for k, v in loss_sums.items()}}


@torch.no_grad()
def validate(
        model: SACON,
        loader: data_utils.DataLoader,
        criterion: SACANLoss,
        device: torch.device,
        writer,
        num_stages: int,
        class_names: List[str],
        ordinal=False,
        fold_save=None,
        every_save_time=None
):
    model.eval()
    total_loss = 0.0

    all_lb_stage_prob, all_lb_stage_true = [], []
    all_rb_stage_prob, all_rb_stage_true = [], []

    loop = tqdm(loader, total=len(loader))
    for batch in loop:
        xl_t1 = batch['left_t1'].to(device)
        xl_t2 = batch['left_t2'].to(device)
        xr_t1 = batch['right_t1'].to(device)
        xr_t2 = batch['right_t2'].to(device)
        labels = {
            'left_binary': batch['left_binary'].to(device),
            'left_stage': batch['label_left_ELS'].to(device),
            'right_binary': batch['right_binary'].to(device),
            'right_stage': batch['label_right_ELS'].to(device),
        }
        with autocast(enabled=True):
            outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)
            loss, _ = criterion(outputs, labels)
            if ordinal:
                outputs["left_stage"] = ordinal_logits_to_class_probs(outputs["left_stage"], num_classes=3)
                outputs["right_stage"] = ordinal_logits_to_class_probs(outputs["right_stage"], num_classes=3)

        total_loss += loss.item()


        all_lb_stage_prob.extend(torch.argmax(torch.softmax(outputs['left_stage'], dim=0), dim=1).cpu().tolist())
        all_lb_stage_true.extend(batch['left_stage'].tolist())
        all_rb_stage_prob.extend(torch.argmax(torch.softmax(outputs['right_stage'], dim=0), dim=1).cpu().tolist())
        all_rb_stage_true.extend(batch['right_stage'].tolist())

    all_lb_stage_true, all_rb_stage_true = np.array(all_lb_stage_true), np.array(all_rb_stage_true)
    all_lb_stage_prob, all_rb_stage_prob = np.array(all_lb_stage_prob), np.array(all_rb_stage_prob)
    other_stage = compute_stage_metrics(np.concatenate((all_lb_stage_true, all_rb_stage_true)),
                                        np.concatenate((all_lb_stage_prob, all_rb_stage_prob)))

    val_loss = total_loss / len(loader)

    return val_loss,other_stage["MCC"]


def run_experiment(
        ablation_name: str,
        train_cfg: TrainConfig,
        trial=None
):
    configs = get_ablation_configs()
    if ablation_name not in configs:
        raise ValueError(f"未知消融配置: {ablation_name}，可选: {list(configs.keys())}")

    ablation_cfg = configs[ablation_name]
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')

    print(f"  消融实验: {ablation_name}")
    print(f"  设备: {device}  | Epochs: {train_cfg.num_epochs}  | BS: {train_cfg.batch_size}")
    print(f"  配置: {vars(ablation_cfg)}")

    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    with open(train_cfg.data_root, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    y = data['label_right_ELS']
    groups = data['patient_id']
    for fold_id, (train_index, test_index) in enumerate(kf.split(data['patient_id'],y,groups)):
        print(train_index, test_index)
        fold_dir = os.path.join('./', f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        train_ds = ONFHDataset(
            data_root=train_cfg.data_root,
            ids=train_index,
            volume_size=train_cfg.volume_size,
            augment=train_cfg.augment
        )
        val_ds = ONFHDataset(
            data_root=train_cfg.data_root,
            ids=test_index,
            volume_size=train_cfg.volume_size
        )
        train_loader = data_utils.DataLoader(
            train_ds, batch_size=train_cfg.batch_size,
            shuffle=True, num_workers=train_cfg.num_workers
        )
        val_loader = data_utils.DataLoader(
            val_ds, batch_size=train_cfg.batch_size,
            shuffle=False, num_workers=train_cfg.num_workers
        )

        model = SACON(ablation_cfg, base_channels=train_cfg.base_channels,
                      num_stages=train_cfg.num_stages,ordinal=train_cfg.ordinal).to(device)
        model.register_cam_hooks()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        stage_class_weights = train_ds.return_stage_class_weights()

        criterion = SACANLoss(
            ablation_cfg,
            alpha_bin=train_cfg.alpha_bin,
            alpha_stage=train_cfg.alpha_stage,
            alpha_reg=train_cfg.alpha_reg,
            alpha_conc=train_cfg.alpha_conc,
            num_stages=train_cfg.num_stages,
            ordinal=train_cfg.ordinal,
            stage_class_weights=stage_class_weights
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.num_epochs
        )

        os.makedirs(train_cfg.save_dir, exist_ok=True)


        train_loop = tqdm(range(1, train_cfg.num_epochs + 1), total=len(range(1, train_cfg.num_epochs + 1)))
        total_step = 0
        writer = None

        for epoch in train_loop:
            total_step += len(train_loader)
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, total_step, writer,
                                            epoch, ordinal=train_cfg.ordinal)
            scheduler.step()

        val_loss, val_mcc = validate(model, val_loader, criterion, device, writer,
                                     train_cfg.num_stages, list(train_cfg.class_names),
                                     ordinal=train_cfg.ordinal, fold_save=fold_dir,
                                     every_save_time=train_cfg.every_save_time)
        return val_loss,val_mcc

def objective(trial):
    parser = argparse.ArgumentParser(description='SACON Training')
    parser.add_argument('--ablation', type=str, default='full',
                        help='消融配置名称（single）或 all（全部运行）')
    parser.add_argument('--data_root', type=str,
                        default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_2026126\onfh_data.json", )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    params = {
        'num_epochs': 30,
        'lr': trial.suggest_float('lr', 0.000001, 0.0001, log=True),
        'alpha_bin':trial.suggest_float('alpha_bin', 0.1, 1.5, log=True),
        'alpha_stage':trial.suggest_float('alpha_stage', 0.1, 1.5, log=True),
        'alpha_reg':trial.suggest_float('alpha_reg', 0.1, 1.0, log=True),
        'alpha_decouple':trial.suggest_float('alpha_decouple', 0.01, 1.0, log=True),
        'ordinal':trial.suggest_categorical('ordinal', [True, False])
    }

    train_cfg = TrainConfig(
        data_root=args.data_root,
        num_stages=args.num_stages,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
        **params
    )

    val_loss,val_mcc = run_experiment(args.ablation, train_cfg,trial=trial)

    return val_loss, val_mcc

def optimize():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study-bin"
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(directions=["minimize", "maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("总试验次数：", len(study.trials))
    print("帕累托最优解集(Pareto front):")
    trials = sorted(study.best_trials, key=lambda t: t.values)
    for trial in trials:
        print("     Trial:", trial.number)
        print(f"    Values: Loss={trial.values[0]}, MCC={trial.values[1]}")
        print("    Params: ", trial.params, '\n')


def visualize():
    study_name = "example-study-bin"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)

    df = study.trials_dataframe()
    print(df)

    print("总试验次数：", len(study.trials))
    print("帕累托最优解集(Pareto front):")
    optuna.visualization.plot_contour(study)
    optuna.visualization.plot_param_importances(study)

if __name__ == '__main__':
    optimize()
