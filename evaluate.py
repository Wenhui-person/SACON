
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.utils.data as data_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.config import TrainConfig, get_ablation_configs
from models.sacon import SACON
from losses.losses import SACANLoss
from data.dataset import DummyDataset
from utils.metrics import compute_binary_metrics, compute_stage_metrics


def _default_binary_metrics() -> dict:
    return {'auc': 0.5, 'accuracy': 0.5, 'sensitivity': 0.5, 'specificity': 0.5, 'f1': 0.}


def _find_checkpoint_path(save_dir: str, ablation_name: str, task: str = '', run_tag: str = '') -> str:
    candidates = []
    if run_tag:
        candidates.extend([
            os.path.join(save_dir, f'{task}_{ablation_name}_{run_tag}_best.pth') if task else '',
            os.path.join(save_dir, f'{ablation_name}_{run_tag}_best.pth'),
        ])
    candidates.extend([
        os.path.join(save_dir, f'{task}_{ablation_name}_best.pth') if task else '',
        os.path.join(save_dir, f'{ablation_name}_best.pth'),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates and candidates[0] else os.path.join(save_dir, f'{ablation_name}_best.pth')


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str,
    val_loader: data_utils.DataLoader,
    device: torch.device,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    ablation_cfg_dict = ckpt.get('ablation_cfg', {})
    train_cfg_dict = ckpt.get('train_cfg', {})

    from configs.config import AblationConfig
    ablation_cfg = AblationConfig(**ablation_cfg_dict)
    base_ch = train_cfg_dict.get('base_channels', 32)
    num_stages = train_cfg_dict.get('num_stages', 5)
    task = train_cfg_dict.get('task', 'bin')
    ordinal = train_cfg_dict.get('ordinal', False)
    run_tag = train_cfg_dict.get('run_tag', '')
    model_variant = ckpt.get('model_variant') or (f"{run_tag}_{os.path.basename(ckpt_path)}" if run_tag else os.path.basename(ckpt_path))

    model = SACON(ablation_cfg, base_channels=base_ch, num_stages=num_stages, ordinal=ordinal, task=task).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_lb_prob, all_lb_true = [], []
    all_rb_prob, all_rb_true = [], []
    all_ls_pred, all_ls_true = [], []
    all_rs_pred, all_rs_true = [], []

    for batch in val_loader:
        xl_t1 = batch['left_t1'].to(device)
        xl_t2 = batch['left_t2'].to(device)
        xr_t1 = batch['right_t1'].to(device)
        xr_t2 = batch['right_t2'].to(device)

        outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)

        if outputs.get('left_binary') is not None:
            all_lb_prob.extend(torch.sigmoid(outputs['left_binary']).squeeze(1).cpu().tolist())
            all_lb_true.extend(batch['left_binary'].tolist())
        if outputs.get('right_binary') is not None:
            all_rb_prob.extend(torch.sigmoid(outputs['right_binary']).squeeze(1).cpu().tolist())
            all_rb_true.extend(batch['right_binary'].tolist())

        if outputs.get('left_stage') is not None:
            all_ls_pred.extend(outputs['left_stage'].argmax(1).cpu().tolist())
            all_ls_true.extend(batch['left_stage'].tolist())
        if outputs.get('right_stage') is not None:
            all_rs_pred.extend(outputs['right_stage'].argmax(1).cpu().tolist())
            all_rs_true.extend(batch['right_stage'].tolist())

    if all_lb_true:
        lb_np = np.array(all_lb_true)
        left_bin_metrics = (compute_binary_metrics(lb_np, np.array(all_lb_prob))
                            if lb_np.sum() > 0 and lb_np.sum() < len(lb_np)
                            else _default_binary_metrics())
    else:
        left_bin_metrics = _default_binary_metrics()

    if all_rb_true:
        rb_np = np.array(all_rb_true)
        right_bin_metrics = (compute_binary_metrics(rb_np, np.array(all_rb_prob))
                             if rb_np.sum() > 0 and rb_np.sum() < len(rb_np)
                             else _default_binary_metrics())
    else:
        right_bin_metrics = _default_binary_metrics()

    left_stage_metrics = compute_stage_metrics(np.array(all_ls_true), np.array(all_ls_pred)) if all_ls_true else {}
    right_stage_metrics = compute_stage_metrics(np.array(all_rs_true), np.array(all_rs_pred)) if all_rs_true else {}

    return {
        'checkpoint_path': ckpt_path,
        'run_tag': run_tag,
        'model_variant': model_variant,
        'left_binary':  left_bin_metrics,
        'right_binary': right_bin_metrics,
        'left_stage':   left_stage_metrics,
        'right_stage':  right_stage_metrics,
        'avg_auc':      (left_bin_metrics['auc'] + right_bin_metrics['auc']) / 2,
    }

def compare_all(
    save_dir: str,
    val_loader: data_utils.DataLoader,
    device: torch.device,
    out_dir: str = './outputs',
    task: str = '',
    run_tag: str = '',
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    configs = get_ablation_configs()
    all_results = {}

    for name in configs:
        ckpt_path = _find_checkpoint_path(save_dir, name, task=task, run_tag=run_tag)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {name}: checkpoint 不存在 ({ckpt_path})")
            continue
        print(f"[EVAL] {name} ...")
        metrics = evaluate_checkpoint(ckpt_path, val_loader, device)
        result_key = metrics.get('model_variant') if metrics.get('run_tag') else name
        result_key = result_key or (f"{run_tag}_{name}" if run_tag else name)
        all_results[result_key] = metrics
        print(
            f"       Avg AUC={metrics['avg_auc']:.4f} | "
            f"L-AUC={metrics['left_binary']['auc']:.4f} | "
            f"R-AUC={metrics['right_binary']['auc']:.4f} | "
            f"L-F1={metrics['left_binary']['f1']:.4f}"
        )

    if not all_results:
        print("无可用结果，请先运行 train.py")
        return {}

    header = f"{'Experiment':<28} {'Avg AUC':>8} {'L-AUC':>7} {'R-AUC':>7} {'L-Sen':>7} {'L-Spe':>7} {'L-F1':>6}"
    print("ABLATION STUDY SUMMARY")
    print(header)

    baseline = all_results.get('no_contralateral', {}).get('avg_auc', None)
    sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['avg_auc'])

    for name, m in sorted_results:
        lb = m['left_binary']
        delta = f" (+{(m['avg_auc']-baseline)*100:.1f}%)" if baseline else ""
        print(
            f"{name:<28} {m['avg_auc']:>8.4f}{delta:<10}"
            f"{lb['auc']:>7.4f} {m['right_binary']['auc']:>7.4f} "
            f"{lb['sensitivity']:>7.4f} {lb['specificity']:>7.4f} {lb['f1']:>6.4f}"
        )

    json_path = os.path.join(out_dir, 'ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump({k: {
            'model_variant': v.get('model_variant', k),
            'run_tag': v.get('run_tag', ''),
            'checkpoint_path': v.get('checkpoint_path', ''),
            'avg_auc': v['avg_auc'],
            'left_auc': v['left_binary']['auc'],
            'right_auc': v['right_binary']['auc'],
            'left_f1': v['left_binary']['f1'],
        } for k, v in all_results.items()}, f, indent=2)
    print(f"\n结果已保存至: {json_path}")

    _plot_ablation_bar(sorted_results, os.path.join(out_dir, 'ablation_bar.png'))

    _plot_training_curves(save_dir, list(all_results.keys()),
                          os.path.join(out_dir, 'training_curves.png'))

    return all_results


def _plot_ablation_bar(sorted_results, save_path):
    names = [r[0] for r in sorted_results]
    aucs  = [r[1]['avg_auc'] for r in sorted_results]

    colors = []
    for n in names:
        if n == 'full':
            colors.append('#2ecc71')
        elif n == 'no_contralateral':
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.3), 5))
    bars = ax.bar(range(len(names)), aucs, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)

    for i, auc in enumerate(aucs):
        ax.text(i, auc + 0.003, f'{auc:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Average AUC (Left + Right)', fontsize=10)
    ax.set_title('SACON Ablation Study — Bilateral AUC Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(max(0, min(aucs) - 0.06), min(1.0, max(aucs) + 0.06))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Full model'),
        Patch(facecolor='#e74c3c', label='Baseline (no contralateral)'),
        Patch(facecolor='#3498db', label='Ablation variant'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] 对比柱状图保存至: {save_path}")


def _plot_training_curves(save_dir, names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — All Ablations', fontsize=13, fontweight='bold')

    plotted = 0
    for name in names:
        hist_path = os.path.join(save_dir, f'{name}_history.json')
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            history = json.load(f)
        epochs  = [h['epoch'] for h in history]
        losses  = [h.get('total', 0) for h in history]
        aucs    = [h.get('avg_auc', 0) for h in history]
        lw = 2.5 if name == 'full' else 1.2
        axes[0].plot(epochs, losses, label=name, linewidth=lw)
        axes[1].plot(epochs, aucs,   label=name, linewidth=lw)
        plotted += 1

    if plotted > 0:
        for ax, ylabel, title in zip(
            axes,
            ['Train Loss', 'Val Avg AUC'],
            ['Training Loss Curves', 'Validation AUC Curves']
        ):
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] 训练曲线保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='SACON Ablation Evaluation')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--out_dir',  type=str, default='./outputs')
    parser.add_argument('--device',   type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--task', type=str, default='',
                        help='optional task prefix in checkpoint names, e.g. stage or bin')
    parser.add_argument('--run_tag', type=str, default='',
                        help='optional checkpoint suffix, e.g. ssl')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    val_ds = DummyDataset(num_samples=100, volume_size=(64, 64, 64))
    val_loader = data_utils.DataLoader(val_ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=0)

    compare_all(args.save_dir, val_loader, device, args.out_dir, task=args.task, run_tag=args.run_tag)


if __name__ == '__main__':
    main()
