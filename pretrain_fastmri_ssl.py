
import argparse
import os
import random
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fastmri_dataset import FastMRIDataset, build_fastmri_manifest, parse_splits
from pretrain_ssl import (
    MaskedReconstructionModel,
    make_block_mask,
    parse_int_tuple,
    reconstruction_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_fastmri_checkpoint(
    model: MaskedReconstructionModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    loss: float,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "modality": "fastmri",
            "encoder_state_dict": model.encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pretrain_cfg": vars(args),
            "train_loss": loss,
            "source_dataset": "fastMRI",
        },
        save_path,
    )
    print(f"[fastMRISSL] Saved checkpoint: {save_path}")


def _middle_slice(volume: torch.Tensor) -> np.ndarray:
    array = volume.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected [D,H,W] or [1,D,H,W], got {array.shape}")
    return array[array.shape[0] // 2]


def save_reconstruction_preview(
    volume: torch.Tensor,
    masked_volume: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    epoch: int,
    loss: float,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    target_slice = _middle_slice(volume[0])
    masked_slice = _middle_slice(masked_volume[0])
    recon_slice = _middle_slice(recon[0])
    mask_slice = _middle_slice(mask[0])
    error_slice = np.abs(recon_slice - target_slice)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    panels = [
        ("Original", target_slice, "gray", 0.0, 1.0),
        ("Masked", masked_slice, "gray", 0.0, 1.0),
        ("Recon", recon_slice, "gray", 0.0, 1.0),
        ("Mask", mask_slice, "gray", 0.0, 1.0),
        ("Abs Error", error_slice, "magma", 0.0, max(1e-6, float(error_slice.max()))),
    ]
    for ax, (title, image, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"fastMRI SSL epoch {epoch} | loss={loss:.6f}", fontsize=11)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_reconstruction.png")
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fastMRISSL] Saved reconstruction preview: {save_path}")


def make_dataset(args: argparse.Namespace, splits: Tuple[str, ...]) -> FastMRIDataset:
    return FastMRIDataset(
        fastmri_root=args.fastmri_root,
        manifest_path=args.manifest_path,
        splits=splits,
        volume_size=args.volume_size,
        max_files=args.max_files,
    )


def train(args: argparse.Namespace, splits: Tuple[str, ...], device: torch.device) -> str:
    dataset = make_dataset(args, splits=splits)
    loader = data_utils.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(
        f"[fastMRISSL] volumes={len(dataset)} batches={len(loader)} "
        f"splits={','.join(splits)} volume_size={args.volume_size}"
    )

    model = MaskedReconstructionModel(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_loss = float("inf")
    best_path = os.path.join(args.out_dir, "fastmri_encoder_ssl_best.pth")
    last_path = os.path.join(args.out_dir, "fastmri_encoder_ssl_last.pth")
    epoch_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        preview = None
        loop = tqdm(loader, total=len(loader), desc=f"fastMRI SSL epoch {epoch}/{args.epochs}")

        for volume in loop:
            volume = volume.to(device, non_blocking=True)
            mask = make_block_mask(volume, args.mask_ratio, args.mask_block_size)
            masked_volume = volume * (1.0 - mask)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and args.amp)):
                recon = model(masked_volume)
                loss = reconstruction_loss(recon, volume, mask, args.full_loss_weight)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})
            if args.vis_interval > 0 and preview is None:
                preview = (
                    volume.detach().cpu(),
                    masked_volume.detach().cpu(),
                    recon.detach().cpu(),
                    mask.detach().cpu(),
                )

        scheduler.step()
        epoch_loss = running_loss / max(1, len(loader))
        print(f"[fastMRISSL] epoch={epoch} loss={epoch_loss:.6f}")

        if args.vis_interval > 0 and preview is not None and epoch % args.vis_interval == 0:
            save_reconstruction_preview(
                volume=preview[0],
                masked_volume=preview[1],
                recon=preview[2],
                mask=preview[3],
                epoch=epoch,
                loss=epoch_loss,
                save_dir=os.path.join(args.out_dir, "recon_vis"),
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_fastmri_checkpoint(model, optimizer, args, epoch, epoch_loss, best_path)

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_fastmri_checkpoint(
                model,
                optimizer,
                args,
                epoch,
                epoch_loss,
                os.path.join(args.out_dir, f"fastmri_encoder_ssl_epoch_{epoch}.pth"),
            )

    save_fastmri_checkpoint(model, optimizer, args, args.epochs, epoch_loss, last_path)
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fastMRI self-supervised encoder pretraining")
    parser.add_argument(
        "--fastmri_root",
        type=str,
        default=r"H:\Training\fastMRI",
        help="Root directory containing fastMRI split folders.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="multicoil_train,singlecoil_train,singlecoil_test,singlecoil_val",
        help="Comma-separated fastMRI split folders to scan.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="CSV manifest path. If omitted, out_dir/fastmri_manifest.csv is created from fastmri_root.",
    )
    parser.add_argument("--build_manifest_only", action="store_true", help="Only scan H5 files and write the manifest.")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/ssl_fastmri")
    parser.add_argument("--max_files", type=int, default=0, help="Limit H5 files for sanity checks. 0 means all.")
    parser.add_argument("--volume_size", type=parse_int_tuple, default=(19, 314, 224))
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--mask_block_size", type=parse_int_tuple, default=(2, 32, 24))
    parser.add_argument("--full_loss_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=5,
        help="Save reconstruction preview every N epochs. Set 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_files = None if args.max_files <= 0 else args.max_files
    splits = parse_splits(args.splits)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.manifest_path:
        args.manifest_path = os.path.join(args.out_dir, "fastmri_manifest.csv")
    if not os.path.exists(args.manifest_path):
        if not args.fastmri_root:
            raise ValueError("--fastmri_root is required to create a missing manifest.")
        rows = build_fastmri_manifest(
            args.fastmri_root,
            args.manifest_path,
            splits=splits,
            max_files=args.max_files,
        )
        print(f"[fastMRISSL] Wrote manifest: {args.manifest_path} rows={len(rows)}")

    if args.build_manifest_only:
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[fastMRISSL] device={device} out_dir={args.out_dir}")
    best_path = train(args, splits=splits, device=device)
    print("[fastMRISSL] Best encoder checkpoint:")
    print(f"  fastmri: {best_path}")


if __name__ == "__main__":
    main()
