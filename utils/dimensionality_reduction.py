import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from models import SACON

try:
    import umap
except ImportError:
    umap = None

try:
    from scipy import stats
except ImportError:
    stats = None


CLASS_NAMES = ("Normal", "Early", "Late")


def _pool_feature(feat: torch.Tensor) -> torch.Tensor:
    if feat is None:
        return None
    if feat.dim() == 5:
        return feat.mean(dim=(2, 3, 4))
    return feat


def _pick_feature_pair(outputs: dict, target: str):
    if target == "z":
        return outputs.get("z_left"), outputs.get("z_right")
    if target == "f":
        return outputs.get("f_left"), outputs.get("f_right")
    if target == "diff_aware":
        feat = outputs.get("diff_aware")
        return feat, feat
    raise ValueError(f"Unsupported feature target: {target}")


@torch.no_grad()
def extract_bilateral_feature_cloud(
        model: SACON,
        dataloader: DataLoader,
        device: str = "cuda",
        target: str = "z",
        task: str = "stage",
) -> Tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()
    features_list = []
    labels_list = []

    for batch in dataloader:
        xl_t1 = batch["left_t1"].to(device)
        xl_t2 = batch["left_t2"].to(device)
        xr_t1 = batch["right_t1"].to(device)
        xr_t2 = batch["right_t2"].to(device)
        outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)

        left_feat, right_feat = _pick_feature_pair(outputs, target)
        left_feat = _pool_feature(left_feat)
        right_feat = _pool_feature(right_feat)
        if left_feat is None or right_feat is None:
            continue

        features_list.extend([left_feat.cpu().numpy(), right_feat.cpu().numpy()])
        if task == "bin":
            labels_list.extend([
                batch["left_binary"].cpu().numpy(),
                batch["right_binary"].cpu().numpy(),
            ])
        else:
            labels_list.extend([
                batch["label_left_ELS"].cpu().numpy(),
                batch["label_right_ELS"].cpu().numpy(),
            ])

    if was_training:
        model.train()

    if not features_list:
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0).astype(int)


def _pca_project(features: np.ndarray, n_components: int):
    if len(features) < 2:
        return None, None
    n_components = min(n_components, features.shape[0], features.shape[1])
    if n_components < 2:
        return None, None
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(features)
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])), mode="constant")
    return projected, pca.explained_variance_ratio_


def _label_name(label: int, class_names=CLASS_NAMES) -> str:
    return class_names[label] if 0 <= int(label) < len(class_names) else str(label)


def plot_feature_cloud_2d(
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        title: str = "Feature Cloud 2D",
        class_names=CLASS_NAMES,
):
    projected, var_ratio = _pca_project(features, 2)
    if projected is None:
        print(f"[FeatureCloud] Skip 2D cloud, not enough feature dimensions: {save_path}")
        return False

    x, y = projected[:, 0], projected[:, 1]
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(np.unique(labels)))))
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, label in enumerate(np.unique(labels)):
        mask = labels == label
        color = colors[int(label) % len(colors)]
        if stats is not None and mask.sum() >= 4:
            try:
                xx, yy = np.mgrid[x.min():x.max():120j, y.min():y.max():120j]
                kde = stats.gaussian_kde(np.vstack([x[mask], y[mask]]))
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=[color], linewidths=0.8, alpha=0.55)
            except Exception as exc:
                print(f"[FeatureCloud] KDE fallback for class {label}: {exc}")
        ax.scatter(
            x[mask], y[mask], s=34, alpha=0.72, color=color,
            edgecolors="white", linewidths=0.5, label=_label_name(label, class_names)
        )

    ratio_text = ""
    if var_ratio is not None and len(var_ratio) >= 2:
        ratio_text = f" ({var_ratio[0] * 100:.1f}%, {var_ratio[1] * 100:.1f}%)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1{ratio_text.split(',')[0] if ratio_text else ''}")
    ax.set_ylabel(f"PC2{(',' + ratio_text.split(',')[1]) if ratio_text else ''}")
    ax.legend(frameon=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[FeatureCloud] Saved 2D cloud: {save_path}")
    return True


def plot_feature_cloud_3d(
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        title: str = "Feature Cloud 3D",
        class_names=CLASS_NAMES,
):
    projected, var_ratio = _pca_project(features, 3)
    if projected is None:
        print(f"[FeatureCloud] Skip 3D cloud, not enough feature dimensions: {save_path}")
        return False

    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(np.unique(labels)))))
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for label in np.unique(labels):
        mask = labels == label
        color = colors[int(label) % len(colors)]
        ax.scatter(
            projected[mask, 0], projected[mask, 1], projected[mask, 2],
            s=28, alpha=0.68, color=color, edgecolors="white", linewidths=0.35,
            label=_label_name(label, class_names)
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(frameon=False, loc="upper left")
    ax.view_init(elev=24, azim=38)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("white")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FeatureCloud] Saved 3D cloud: {save_path}")
    return True


def save_feature_clouds(
        model: SACON,
        dataloader: DataLoader,
        device: str,
        save_dir: str,
        prefix: str,
        target: str = "z",
        task: str = "stage",
        save_3d: bool = True,
        class_names=CLASS_NAMES,
):
    try:
        features, labels = extract_bilateral_feature_cloud(model, dataloader, device, target, task)
        if features.shape[0] < 2:
            print(f"[FeatureCloud] Skip {prefix}, no enough samples.")
            return
        plot_feature_cloud_2d(
            features, labels, os.path.join(save_dir, f"feature_cloud_2d_{prefix}.png"),
            title=f"Feature Cloud 2D ({prefix})", class_names=class_names,
        )
        if save_3d:
            plot_feature_cloud_3d(
                features, labels, os.path.join(save_dir, f"feature_cloud_3d_{prefix}.png"),
                title=f"Feature Cloud 3D ({prefix})", class_names=class_names,
            )
    except Exception as exc:
        print(f"[FeatureCloud] Failed to save {prefix}: {exc}")


def extract_features_from_sacon(
        model: SACON,
        dataloader: DataLoader,
        device: str = 'cuda',
        feature_type: str = 'z_left'
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            xl_t1 = batch['left_t1'].to(device)
            xl_t2 = batch['left_t2'].to(device)
            xr_t1 = batch['right_t1'].to(device)
            xr_t2 = batch['right_t2'].to(device)

            labels = batch['left_binary'].cpu().numpy()

            outputs = model(xl_t1, xl_t2, xr_t1, xr_t2)

            feat = outputs[feature_type]

            if feat.dim() == 5:
                feat = feat.mean(dim=[2, 3, 4])

            features_list.append(feat.cpu().numpy())
            labels_list.append(labels)

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels

def visualize_pca_with_density(features, labels, save_path='pca_visualization.png'):
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)

    var_ratio = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(10, 10))

    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                          height_ratios=[1, 4, 0.1], width_ratios=[0.1, 4, 1])

    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    label_names = ['Normal','Early','Late']
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax_main.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=[colors[i]], alpha=0.6, s=50, label=f'{label_names[label]}')

    ax_main.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)', fontsize=12)
    ax_main.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)', fontsize=12)
    ax_main.legend(loc='best', frameon=False)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        density = stats.gaussian_kde(features_2d[mask, 0])
        xs = np.linspace(features_2d[:, 0].min(), features_2d[:, 0].max(), 200)
        ax_top.fill_between(xs, density(xs), alpha=0.6, color=colors[i])

    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        density = stats.gaussian_kde(features_2d[mask, 1])
        ys = np.linspace(features_2d[:, 1].min(), features_2d[:, 1].max(), 200)
        ax_right.fill_betweenx(ys, density(ys), alpha=0.6, color=colors[i])

    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"PCA可视化已保存至: {save_path}")


def visualize_umap(features, labels, save_path='umap_visualization.png'):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    features_2d = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"UMAP可视化已保存至: {save_path}")


def visualize_tsne(features, labels, save_path='tsne_visualization.png'):
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"t-SNE可视化已保存至: {save_path}")



if __name__ == '__main__':

    pass
