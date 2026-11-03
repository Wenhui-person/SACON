
import torch
import torch.nn as nn
from typing import Dict, Optional
import torch.nn.functional as F

from configs.config import AblationConfig
from .alignment import SymmetryAlignmentModule
from .encoders import ModalityEncoder3D
from .fusion import CrossModalGatedFusion
from .interaction import ContralateralInteractionModule
from .heads import MultiTaskHead,BinaryHead,StageHead


class LocalDiffGatedFusion(nn.Module):
    def __init__(self, local_channels: int, diff_channels: int, out_channels: int, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate

        self.local_proj = nn.Conv3d(local_channels, out_channels, kernel_size=1, bias=False)
        self.diff_proj = nn.Conv3d(diff_channels, out_channels, kernel_size=1, bias=False)

        self.gate = nn.Sequential(
            nn.Conv3d(local_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_local: torch.Tensor, z_diff: torch.Tensor):
        local = self.local_proj(z_local)
        diff = self.diff_proj(z_diff)
        gate = self.gate(z_local)

        fused = local + gate * diff if self.use_gate else local + diff
        fused = self.act(self.norm(fused))
        return fused, gate

class SACON(nn.Module):

    def __init__(
        self,
        config: AblationConfig,
        base_channels: int = 32,
        num_stages: int = 3,
        ordinal=None,
        task=None
    ):
        super().__init__()
        self.config = config

        self.task = task

        self.alignment = SymmetryAlignmentModule(config)

        self.t1_encoder = ModalityEncoder3D(in_channels=1, base_channels=base_channels)
        self.t2_encoder = ModalityEncoder3D(in_channels=1, base_channels=base_channels)
        feat_ch = self.t1_encoder.out_channels

        self.modal_fusion = CrossModalGatedFusion(feat_ch, use_gating=config.use_gated_fusion)

        self.contralateral = ContralateralInteractionModule(feat_ch, config)

        if not config.use_contralateral:
            final_ch = feat_ch
        elif not config.use_explicit_difference:
            final_ch = feat_ch * 2
        elif config.use_contralateral:
            if config.use_structure_signal_decoupling:
                final_ch = feat_ch * 4
            else:
                final_ch = feat_ch*3

        local_ch = feat_ch * 2
        diff_ch = feat_ch * 2

        self.left_fusion_gate = LocalDiffGatedFusion(
            local_channels=local_ch,
            diff_channels=diff_ch,
            out_channels=final_ch,
            use_gate=config.use_local_diff_gate,
        )
        self.right_fusion_gate = LocalDiffGatedFusion(
            local_channels=local_ch,
            diff_channels=diff_ch,
            out_channels=final_ch,
            use_gate=config.use_local_diff_gate,
        )


        self.left_bin_head = BinaryHead(in_channels=final_ch)
        self.right_bin_head = BinaryHead(in_channels=final_ch)

        num_stages = num_stages if ordinal else 4

        self.left_stage_head = StageHead(in_channels=final_ch, num_stages=num_stages)
        self.right_stage_head = StageHead(in_channels=final_ch, num_stages=num_stages)

        self._init_module(self.t1_encoder)
        self._init_module(self.t2_encoder)
        self._init_module(self.modal_fusion)
        self._init_module(self.contralateral)
        self._feat_maps: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def _init_module(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def register_cam_hooks(self):
        self._feat_maps.clear()
        self._gradients.clear()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        def _fwd_hook(name):
            def hook(module, inp, out):
                self._feat_maps[name] = out.detach()

            return hook

        def _bwd_hook(name):
            def hook(module, grad_in, grad_out):
                if grad_out is not None and len(grad_out) > 0:
                    self._gradients[name] = grad_out[0].detach()
                else:
                    self._gradients[name] = None

            return hook

        targets = {
            't1_layer3': self.t1_encoder.layer3_conv,
            't2_layer3': self.t2_encoder.layer3_conv,
            'left_stage_head_input': self.left_stage_head.proj[0] if hasattr(self.left_stage_head, 'proj') and len(self.left_stage_head.proj) > 0 else None,
            'right_stage_head_input': self.right_stage_head.proj[0] if hasattr(self.right_stage_head, 'proj') and len(self.right_stage_head.proj) > 0 else None,
        }

        for name, module in targets.items():
            if module is not None:
                self._hooks.append(module.register_forward_hook(_fwd_hook(name)))
                self._hooks.append(module.register_full_backward_hook(_bwd_hook(name)))

        for name in ['z_left', 'z_right']:
            self._feat_maps[name] = None
            self._gradients[name] = None

    def remove_cam_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(
        self,
        xl_t1: torch.Tensor,
        xl_t2: torch.Tensor,
        xr_t1: torch.Tensor,
        xr_t2: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        xl_t1, xl_t2, xr_t1_a, xr_t2_a, disp_fields = \
            self.alignment(xl_t1, xl_t2, xr_t1, xr_t2)

        def apply_roi_mask(img_t1, img_t2):
            threshold = img_t1.mean(dim=[2, 3, 4], keepdim=True) * 0.35
            mask = (img_t1 > threshold).float()
            return img_t1 * mask, img_t2 * mask
        xl_t1, xl_t2 = apply_roi_mask(xl_t1, xl_t2)
        xr_t1_a, xr_t2_a = apply_roi_mask(xr_t1_a, xr_t2_a)

        fl_t1 = self.t1_encoder(xl_t1)
        fl_t2 = self.t2_encoder(xl_t2)
        fr_t1 = self.t1_encoder(xr_t1_a)
        fr_t2 = self.t2_encoder(xr_t2_a)

        f_left, modal_gate_left = self.modal_fusion(fl_t1, fl_t2, return_gate=True)
        f_right, modal_gate_right = self.modal_fusion(fr_t1, fr_t2, return_gate=True)

        def retain_hip_core(feat_left: torch.Tensor, feat_right: torch.Tensor, strength: float = 0.65):
            avg_left = feat_left.mean(dim=1, keepdim=True)
            avg_right = feat_right.mean(dim=1, keepdim=True)

            B, _, D, H, W = avg_left.shape
            d_coords = torch.linspace(-1, 1, D, device=feat_left.device).view(1, 1, D, 1, 1)
            h_coords = torch.linspace(-1, 1, H, device=feat_left.device).view(1, 1, 1, H, 1)
            w_coords = torch.linspace(-1, 1, W, device=feat_left.device).view(1, 1, 1, 1, W)

            center_weight = torch.exp(-2.0 * (d_coords ** 2 + h_coords ** 2 + w_coords ** 2))
            center_weight = center_weight / center_weight.max()

            def spatial_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
                B = x.shape[0]
                x_flat = x.reshape(B, -1).float()
                quant = torch.quantile(x_flat, q, dim=1)
                return quant.view(B, 1, 1, 1, 1)

            thresh_left = spatial_quantile(avg_left, 0.35)
            thresh_right = spatial_quantile(avg_right, 0.35)

            mask_left = torch.sigmoid((avg_left - thresh_left) * 4.0)
            mask_left = mask_left * center_weight
            mask_left = mask_left ** strength

            mask_right = torch.sigmoid((avg_right - thresh_right) * 4.0)
            mask_right = mask_right * center_weight
            mask_right = mask_right ** strength

            mask_left = F.avg_pool3d(mask_left, kernel_size=3, stride=1, padding=1)
            mask_right = F.avg_pool3d(mask_right, kernel_size=3, stride=1, padding=1)

            return feat_left * mask_left, feat_right * mask_right


        f_left, f_right = retain_hip_core(f_left, f_right, strength=0.4)

        diff_feat, diff_aware, common_feat, diff_abs,(left_str, right_str), (left_sig, right_sig) = \
            self.contralateral(f_left, f_right)

        z_left_local = z_right_local = None
        z_left_diff = z_right_diff = None
        left_diff_gate = right_diff_gate = None

        if self.config.use_contralateral and not self.config.use_explicit_difference:
            z_left = torch.cat([f_left, f_right], dim=1)
            z_right = torch.cat([f_right, f_left], dim=1)
            diff_feat = diff_aware = diff_abs = None
        elif self.config.use_contralateral:
            if left_str is not None and right_str is not None:
                z_left_local = torch.cat([left_str, left_sig], dim=1)
                z_left_diff = torch.cat([diff_feat, diff_aware], dim=1)

                z_right_local = torch.cat([right_str, right_sig], dim=1)
                z_right_diff = torch.cat([-diff_feat, diff_aware], dim=1)

                z_left, left_diff_gate = self.left_fusion_gate(z_left_local, z_left_diff)
                z_right, right_diff_gate = self.right_fusion_gate(z_right_local, z_right_diff)

            else:
                z_left = torch.cat([f_left, diff_feat, diff_aware], dim=1)
                z_right = torch.cat([f_right,-diff_feat, diff_aware], dim=1)
        else:
            z_left = f_left
            z_right = f_right

        z_left = z_left.requires_grad_(True)
        z_right = z_right.requires_grad_(True)

        self._feat_maps['z_left'] = z_left
        self._feat_maps['z_right'] = z_right

        if self.task == 'bin':
            left_bin = self.left_bin_head(z_left)
            right_bin = self.right_bin_head(z_right)
            return {
            'left_binary': left_bin,
            'right_binary': right_bin,
            'f_left': f_left,
            'f_right': f_right,
            'f_left_str': left_str,
            'f_right_str': right_str,
            'f_left_sig': left_sig,
            'f_right_sig': right_sig,

            'diff_feat': diff_feat,
            'diff_aware': diff_aware,
            'diff_abs': diff_abs,
            'modal_gate_left': modal_gate_left,
            'modal_gate_right': modal_gate_right,
            'z_left': z_left,
            'z_right': z_right,
            'disp_fields': disp_fields,

            'xl_t1_aligned': xl_t1,
            'xl_t2_aligned': xl_t2,
            'xr_t1_aligned': xr_t1_a,
            'xr_t2_aligned': xr_t2_a,

            'z_left_local': z_left_local,
            'z_right_local': z_right_local,
            'z_left_diff': z_left_diff,
            'z_right_diff': z_right_diff,
        }

        if self.task == 'stage':
            left_stage = self.left_stage_head(z_left)
            right_stage = self.right_stage_head(z_right)

            return {
                'left_stage': left_stage,
                'right_stage': right_stage,
                'f_left': f_left,
                'f_right': f_right,

                'f_left_str': left_str,
                'f_right_str': right_str,
                'f_left_sig': left_sig,
                'f_right_sig': right_sig,

                'diff_feat': diff_feat,
                'diff_aware': diff_aware,
                'diff_abs': diff_abs,
                'modal_gate_left': modal_gate_left,
                'modal_gate_right': modal_gate_right,
                'z_left': z_left,
                'z_right': z_right,
                'disp_fields': disp_fields,

                'xl_t1_aligned': xl_t1,
                'xl_t2_aligned': xl_t2,
                'xr_t1_aligned': xr_t1_a,
                'xr_t2_aligned': xr_t2_a,

                'z_left_local': z_left_local,
                'z_right_local': z_right_local,
                'z_left_diff': z_left_diff,
                'z_right_diff': z_right_diff,
                'left_diff_gate': left_diff_gate,
                'right_diff_gate': right_diff_gate
            }
