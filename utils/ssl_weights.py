
from typing import Dict, Optional

import torch


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    for key in ("encoder_state_dict", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise KeyError("Checkpoint must contain encoder_state_dict, model_state_dict, state_dict, or raw tensor weights.")


def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):] if key.startswith(prefix) else key


def _prepare_encoder_state(state_dict: Dict[str, torch.Tensor], encoder_name: str) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = _strip_prefix(key, "module.")
        key = _strip_prefix(key, "model.")
        cleaned[key] = value

    encoder_prefix = f"{encoder_name}."
    if any(key.startswith(encoder_prefix) for key in cleaned):
        return {
            key[len(encoder_prefix):]: value
            for key, value in cleaned.items()
            if key.startswith(encoder_prefix)
        }

    for prefix in ("encoder.", "backbone."):
        if any(key.startswith(prefix) for key in cleaned):
            return {
                key[len(prefix):]: value
                for key, value in cleaned.items()
                if key.startswith(prefix)
            }

    return cleaned


def _load_one_encoder(model, encoder_attr: str, ckpt_path: str, device, strict: bool) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    encoder_state = _prepare_encoder_state(state_dict, encoder_attr)
    encoder = getattr(model, encoder_attr)
    incompatible = encoder.load_state_dict(encoder_state, strict=strict)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(
        f"[SSL] Loaded {encoder_attr} from {ckpt_path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[SSL]   missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[SSL]   unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    return {
        "path": ckpt_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def load_ssl_encoder_weights(
    model,
    t1_ckpt: Optional[str] = None,
    t2_ckpt: Optional[str] = None,
    device=None,
    strict: bool = False,
) -> dict:
    loaded = {}
    if t1_ckpt:
        loaded["t1_encoder"] = _load_one_encoder(model, "t1_encoder", t1_ckpt, device, strict)
    if t2_ckpt:
        loaded["t2_encoder"] = _load_one_encoder(model, "t2_encoder", t2_ckpt, device, strict)

    if not loaded:
        print("[SSL] No SSL encoder checkpoints provided; using random initialization.")
    return loaded


def set_sacon_encoder_trainable(model, trainable: bool) -> None:
    for encoder_attr in ("t1_encoder", "t2_encoder"):
        encoder = getattr(model, encoder_attr, None)
        if encoder is None:
            continue
        for param in encoder.parameters():
            param.requires_grad = trainable
