
import os.path

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    f1_score, cohen_kappa_score, classification_report, average_precision_score, precision_recall_curve, roc_curve, auc,
    brier_score_loss,matthews_corrcoef
)
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import torch
import torch.nn.functional as F


def ordinal_logits_to_class_probs(logits, num_classes):
    if isinstance(logits, torch.Tensor):
        q = torch.sigmoid(logits.detach()).cpu().numpy()
    else:
        q = 1.0 / (1.0 + np.exp(-np.asarray(logits)))

    q = np.minimum.accumulate(q, axis=1)

    B, C = q.shape
    if C != num_classes - 1:
        raise ValueError(
            f"Expected ordinal logits shape (B, {num_classes - 1}), got {q.shape}"
        )

    probs = np.zeros((B, num_classes), dtype=np.float32)

    probs[:, 0] = 1.0 - q[:, 0]

    for c in range(1, num_classes - 1):
        probs[:, c] = q[:, c - 1] - q[:, c]

    probs[:, num_classes - 1] = q[:, -1]

    probs = np.clip(probs, 0.0, 1.0)

    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    probs = torch.from_numpy(probs).float()
    return probs

def plot_class_distribution(
        class_counts: Dict[int, int],
        class_names: List[str],
        save_path: str,
        title: str = "Class Distribution",
):
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(class_names, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=class_names, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Class Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    return {
        'auc':         roc_auc_score(y_true, y_prob),
        'accuracy':    accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'ppv':         ppv,
        'npv':         npv,
    }


def compute_stage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred,),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'kappa':    cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def aggregate_bilateral_metrics(
    left_metrics: Dict[str, float],
    right_metrics: Dict[str, float]
) -> Dict[str, float]:
    keys = set(left_metrics) & set(right_metrics)
    avg = {k: (left_metrics[k] + right_metrics[k]) / 2 for k in keys}
    avg['left_auc'] = left_metrics.get('auc', 0)
    avg['right_auc'] = right_metrics.get('auc', 0)
    return avg

class MetricTracker:

    def __init__(self, num_classes: int, class_names: List[str],ordinal=True):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
        self.ordinal = ordinal

    def reset(self):
        self.all_preds = []
        self.all_probs = []
        self.all_labels = []
        self.all_ids = []
        self.total_loss = 0.0
        self.num_samples = 0

    def update(
            self,
            loss: float,
            logits: torch.Tensor,
            labels: torch.Tensor,
            batch_size: int,
            sample_ids=None,
    ):
        if self.ordinal:
            probs = ordinal_logits_to_class_probs(logits, self.num_classes)
            preds = (torch.sigmoid(logits.detach()) >= 0.5).sum(dim=1).cpu()
        else:
            probs = torch.softmax(logits.detach(), dim=1)
            preds = torch.argmax(probs, dim=1)

        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_probs.extend(probs.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        if sample_ids is None:
            self.all_ids.extend([""] * batch_size)
        else:
            self.all_ids.extend([str(sample_id) for sample_id in sample_ids])
        self.total_loss += loss * batch_size
        self.num_samples += batch_size

    def compute(self,fold_id,epoch) -> Dict[str, float]:
        if self.num_samples == 0:
            return {}

        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)

        save_origin_data = {'fold_id':np.array([fold_id]*len(self.all_labels)),'epoch':np.array([epoch]*len(self.all_labels)),'patient_id': np.array(self.all_ids),'labels': labels, 'preds': preds, 'probs1': probs[:,0],'probs2': probs[:,1],'probs3': probs[:,2]}
        pd.DataFrame(save_origin_data).to_csv('fold_stage_origin_data.csv', mode='a',index=False)

        metrics = {}

        metrics["loss"] = self.total_loss / self.num_samples
        metrics["accuracy"] = float((preds == labels).mean())

        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))

        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        metrics["mean_class_accuracy"] = float(per_class_acc.mean())
        for i, name in enumerate(self.class_names):
            metrics[f"acc_{name}"] = float(per_class_acc[i])

        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics["precision_macro"] = report["macro avg"]["precision"]
        metrics["recall_macro"] = report["macro avg"]["recall"]
        metrics["f1_macro"] = report["macro avg"]["f1-score"]
        metrics["precision_weighted"] = report["weighted avg"]["precision"]
        metrics["recall_weighted"] = report["weighted avg"]["recall"]
        metrics["f1_weighted"] = report["weighted avg"]["f1-score"]

        for name in self.class_names:
            if name in report:
                metrics[f"f1_{name}"] = report[name]["f1-score"]

        metrics["kappa"] = cohen_kappa_score(labels, preds)

        try:
            probs_normed = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
            if self.num_classes == 2:
                metrics["auc_roc"] = roc_auc_score(labels, probs_normed[:, 1])
            else:
                metrics["auc_roc"] = roc_auc_score(
                    labels, probs_normed,
                    multi_class="ovr",
                    average="macro",
                )
        except Exception:
            metrics["auc_roc"] = 0.0
        try:
            ap_list = []
            for c in range(self.num_classes):
                binary_labels = (labels == c).astype(int)
                ap = average_precision_score(binary_labels, probs[:, c])
                ap_list.append(ap)
                metrics[f"ap_{self.class_names[c]}"] = ap
            metrics["mAP"] = float(np.mean(ap_list))
        except Exception:
            metrics["mAP"] = 0.0

        metrics["MCC"] = matthews_corrcoef(labels, preds)

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        return confusion_matrix(labels, preds, labels=list(range(self.num_classes)))

    def get_probabilities(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.all_labels), np.array(self.all_probs)
def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        epoch=None,
        fold_dir=None
):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / (row_sums + 1e-8)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()
    pd.DataFrame(cm_plot).to_csv(fold_dir +"/stage_confusion.csv", mode='a',index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
        labels: np.ndarray,
        probs: np.ndarray,
        class_names: List[str],
        save_path: str,
        epoch=None,
        fold_dir=None
):
    num_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, ax = plt.subplots(figsize=(8, 7))

    auc_scores = []
    for i, (name, color) in enumerate(zip(class_names, colors)):
        binary = (labels == i).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        auc = roc_auc_score(binary, probs[:, i])
        auc_scores.append(auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        pd.DataFrame({"Epoch": [epoch]*len(fpr), "fpr": fpr, "tpr": tpr}).to_csv(fold_dir +'/'+ "stage_ROC.csv",
                                                                                          mode='a', index=False,
                                                                                          header=False)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curves (Macro AUC={np.mean(auc_scores):.3f})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
        history: Dict[str, List],
        save_path: str,
):
    keys_to_plot = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("auc_roc", "AUC-ROC"),
        ("f1_macro", "F1 (Macro)"),
    ]

    available = [(k, label) for k, label in keys_to_plot
                 if f"train_{k}" in history or f"val_{k}" in history]

    n = len(available)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (key, label) in zip(axes, available):
        epochs = range(1, len(history.get(f"train_{key}", [])) + 1)
        if f"train_{key}" in history:
            ax.plot(epochs, history[f"train_{key}"],
                    "b-o", markersize=3, label="Train", lw=2)
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"],
                    "r-o", markersize=3, label="Val", lw=2)

        if f"val_{key}" in history:
            vals = history[f"val_{key}"]
            if key == "loss":
                best_ep = int(np.argmin(vals)) + 1
                best_val = min(vals)
            else:
                best_ep = int(np.argmax(vals)) + 1
                best_val = max(vals)
            ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.6)
            ax.annotate(
                f"Best: {best_val:.4f}",
                xy=(best_ep, best_val),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="red",
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Training History", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curves(
        labels: np.ndarray,
        probs: np.ndarray,
        class_names: List[str],
        save_path: str,
        epoch=None,
        fold_dir=None
):
    num_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        binary = (labels == i).astype(int)
        if binary.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary, probs[:, i])
        ap = average_precision_score(binary, probs[:, i])
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
        pd.DataFrame({"Epoch":[epoch]*len(recall),"recall": recall, "precision": precision}).to_csv(fold_dir+'/'+"stage_PR.csv", mode='a',index=False, header=False)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
        class_counts: Dict[int, int],
        class_names: List[str],
        save_path: str,
        title: str = "Class Distribution",
):
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(class_names, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=class_names, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Class Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binary_classification_metrics(y_true, y_pred, y_prob=None, save_path=None,epoch=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cm = confusion_matrix(y_true, y_pred,labels=[0,1])
    pd.DataFrame(cm).to_csv(os.path.join(save_path,'binary_confusion_matrix.csv'), mode='a',index=False)
    im = axes[0].imshow(cm, cmap='YlGnBu')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')

    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=20)

    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    plt.colorbar(im, ax=axes[0])

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pd.DataFrame({'epoch': [epoch]*len(fpr), 'fpr': fpr, 'tpr': tpr, 'AUC': [roc_auc]*len(fpr)}).to_csv(
            os.path.join(save_path, 'binary_ROC.csv'), mode='a',index=False, header=True)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('FPR')
        axes[1].set_ylabel('TPR')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc='lower right')
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, '需要预测概率', ha='center', va='center')
        axes[1].set_title('ROC Curve')

    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        pd.DataFrame({'epoch': [epoch]*len(precision), 'precision': precision, 'recall': recall,'AP':[pr_auc]*len(precision)}).to_csv(os.path.join(save_path, 'binary_PR.csv'),mode='a', index=False, header=True)

        axes[2].plot(recall, precision, color='green', lw=2,
                     label=f'PR (AP = {pr_auc:.3f})')
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('PR Curve')
        axes[2].legend(loc='lower left')
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, '需要预测概率', ha='center', va='center')
        axes[2].set_title('PR Curve')

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path,'binary_classification.png'), dpi=300, bbox_inches='tight')

    plt.close(fig)

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常', '坏死'],zero_division=0))

def dca_curve(y_true, y_prob, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    N = len(y_true)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    nb_model = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb = (tp / N) - (fp / N) * (pt / (1 - pt + 1e-12))
        nb_model.append(nb)
    nb_model = np.array(nb_model)

    prev = y_true.mean()
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds + 1e-12))
    nb_none = np.zeros_like(thresholds)
    return thresholds, nb_model, nb_all, nb_none

def plot_dca(thresholds, nb_model, nb_all, nb_none, title="Decision Curve",path=None):
    fig = plt.figure(figsize=(5.0, 3.6), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(thresholds, nb_model, label="Model")
    ax.plot(thresholds, nb_all, label="Treat All")
    ax.plot(thresholds, nb_none, label="Treat None")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim(min(nb_model.min(), nb_all.min(), -0.02), max(nb_model.max(), 0.8))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    return fig


def calibration_metrics(y_true, y_prob, n_bins=10, strategy="uniform"):

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    brier = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        m = (ids == b)
        if not m.any():
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier": float(brier),
        "ece": float(ece),
    }

def plot_calibration(prob_true, prob_pred, title="Calibration Curve",path=None):
    fig = plt.figure(figsize=(4.2, 3.6), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    return fig

