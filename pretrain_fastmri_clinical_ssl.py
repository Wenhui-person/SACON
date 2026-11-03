
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

from data.fastmri_clinical_dataset import (
    DEFAULT_CLINICAL_ROOTS,
    FastMRIClinicalDataset,
    build_fastmri_clinical_manifest,
    parse_clinical_roots,
    parse_filter,
)
from models.encoders import ModalityEncoder3D
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


def assert_sacon_encoder_compatible(model: MaskedReconstructionModel, base_channels: int) -> None:
    reference = ModalityEncoder3D(in_channels=1, base_channels=base_channels)
    model_state = model.encoder.state_dict()
    reference_state = reference.state_dict()
    if set(model_state.keys()) != set(reference_state.keys()):
        missing = sorted(set(reference_state.keys()) - set(model_state.keys()))
        unexpected = sorted(set(model_state.keys()) - set(reference_state.keys()))
        raise RuntimeError(
            "SSL encoder keys do not match SACON ModalityEncoder3D. "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    shape_mismatch = [
        (key, tuple(model_state[key].shape), tuple(reference_state[key].shape))
        for key in model_state
        if tuple(model_state[key].shape) != tuple(reference_state[key].shape)
    ]
    if shape_mismatch:
        raise RuntimeError(f"SSL encoder shapes do not match SACON ModalityEncoder3D: {shape_mismatch[:5]}")

    print(
        "[fastMRIClinicalSSL] Encoder compatibility OK: "
        f"ModalityEncoder3D(in_channels=1, base_channels={base_channels}, "
        f"out_channels={reference.out_channels})"
    )


def save_checkpoint(
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
            "modality": "fastmri_clinical",
            "encoder_state_dict": model.encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pretrain_cfg": vars(args),
            "train_loss": loss,
            "source_dataset": "fastMRI clinical DICOM",
            "encoder_class": "ModalityEncoder3D",
            "base_channels": args.base_channels,
            "in_channels": 1,
            "out_channels": args.base_channels * 8,
        },
        save_path,
    )
    print(f"[fastMRIClinicalSSL] Saved checkpoint: {save_path}")


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
    fig.suptitle(f"fastMRI clinical SSL epoch {epoch} | loss={loss:.6f}", fontsize=11)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_reconstruction.png")
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fastMRIClinicalSSL] Saved reconstruction preview: {save_path}")


def make_dataset(args: argparse.Namespace, clinical_roots: Tuple[str, ...]) -> FastMRIClinicalDataset:
    return FastMRIClinicalDataset(
        clinical_roots=clinical_roots,
        manifest_path=args.manifest_path,
        volume_size=args.volume_size,
        max_series=args.max_series,
        include_planes=parse_filter(args.include_planes),
        include_contrasts=parse_filter(args.include_contrasts),
    )


def train(args: argparse.Namespace, clinical_roots: Tuple[str, ...], device: torch.device) -> str:
    dataset = make_dataset(args, clinical_roots=clinical_roots)
    loader = data_utils.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(
        f"[fastMRIClinicalSSL] series={len(dataset)} batches={len(loader)} "
        f"volume_size={args.volume_size}"
    )

    model = MaskedReconstructionModel(base_channels=args.base_channels).to(device)
    assert_sacon_encoder_compatible(model, args.base_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_loss = float("inf")
    best_path = os.path.join(args.out_dir, "fastmri_clinical_encoder_ssl_best.pth")
    last_path = os.path.join(args.out_dir, "fastmri_clinical_encoder_ssl_last.pth")
    epoch_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        preview = None
        loop = tqdm(loader, total=len(loader), desc=f"fastMRI clinical SSL epoch {epoch}/{args.epochs}")

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
        print(f"[fastMRIClinicalSSL] epoch={epoch} loss={epoch_loss:.6f}")

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
            save_checkpoint(model, optimizer, args, epoch, epoch_loss, best_path)

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                args,
                epoch,
                epoch_loss,
                os.path.join(args.out_dir, f"fastmri_clinical_encoder_ssl_epoch_{epoch}.pth"),
            )

    save_checkpoint(model, optimizer, args, args.epochs, epoch_loss, last_path)
    return best_path


def parse_args() -> argparse.Namespace:
    default_roots = os.pathsep.join(DEFAULT_CLINICAL_ROOTS)
    parser = argparse.ArgumentParser(description="fastMRI clinical DICOM self-supervised encoder pretraining")
    parser.add_argument(
        "--clinical_roots",
        type=str,
        default="H:\Training\knee _mri_clinical",
        help="Clinical DICOM roots separated by the OS path separator.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="CSV manifest path. If omitted, out_dir/fastmri_clinical_manifest.csv is created.",
    )
    parser.add_argument("--build_manifest_only", action="store_true", help="Only scan DICOM series and write manifest.")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/ssl_fastmri_clinical")
    parser.add_argument("--max_series", type=int, default=0, help="Limit DICOM series for sanity checks. 0 means all.")
    parser.add_argument("--include_planes", type=str, default="", help="Optional comma-separated planes, e.g. AX,COR,SAG.")
    parser.add_argument(
        "--include_contrasts",
        type=str,
        default="",
        help="Optional comma-separated contrast groups, e.g. PD,PD_FS,T2_FS.",
    )
    parser.add_argument("--volume_size", type=parse_int_tuple, default=(19, 314, 224))
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
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
    args.max_series = None if args.max_series <= 0 else args.max_series
    clinical_roots = parse_clinical_roots(args.clinical_roots)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.manifest_path:
        args.manifest_path = os.path.join(args.out_dir, "fastmri_clinical_manifest.csv")
    if not os.path.exists(args.manifest_path):
        rows = build_fastmri_clinical_manifest(
            clinical_roots=clinical_roots,
            manifest_path=args.manifest_path,
            max_series=args.max_series,
        )
        print(f"[fastMRIClinicalSSL] Wrote manifest: {args.manifest_path} rows={len(rows)}")

    if args.build_manifest_only:
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[fastMRIClinicalSSL] device={device} out_dir={args.out_dir}")
    best_path = train(args, clinical_roots=clinical_roots, device=device)
    print("[fastMRIClinicalSSL] Best encoder checkpoint:")
    print(f"  fastmri_clinical: {best_path}")


if __name__ == "__main__":
    main()
