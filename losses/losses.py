
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple,List
from configs.config import AblationConfig

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()

        self.pos_weight = pos_weight


    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight
        )


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()

class OrdinalStageLoss(nn.Module):
    def __init__(
        self,
        num_stages: int = 5,
        ordinal: bool = False,
        label_smoothing: float = 0.1,
        stage_class_weight: torch.Tensor = None
    ):
        super().__init__()
        self.num_stages = num_stages
        self.ordinal = ordinal
        self.label_smoothing = label_smoothing

        if stage_class_weight is not None:
            self.register_buffer("stage_class_weight", stage_class_weight.float())
        else:
            self.stage_class_weight = None

        if not ordinal:
            self.ce = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=self.stage_class_weight.to("cuda")
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.ordinal:
            return self.ce(logits, targets)

        B, C = logits.shape
        expected_c = self.num_stages - 1
        if C != expected_c:
            raise ValueError(
                f"Ordinal mode expects logits shape (B, {expected_c}), got {tuple(logits.shape)}"
            )

        levels = torch.arange(self.num_stages - 1, device=targets.device)
        ordinal_labels = (targets.unsqueeze(1) > levels.unsqueeze(0)).float()

        loss_mat = F.binary_cross_entropy_with_logits(
            logits,
            ordinal_labels,
            reduction="none",

        )

        loss_per_sample = loss_mat.mean(dim=1)

        if self.stage_class_weight is not None:
            sample_weights = self.stage_class_weight.to(targets.device)[targets]
            loss_per_sample = loss_per_sample * sample_weights
        return loss_per_sample.mean()

class RegistrationLoss(nn.Module):
    def __init__(
        self,
        lambda_ncc: float = 1.0,
        lambda_smooth: float = 0.1
    ):
        super().__init__()
        self.lambda_ncc = lambda_ncc
        self.lambda_smooth = lambda_smooth

    @staticmethod
    def ncc_loss(fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        if fixed.ndim != 5 or warped.ndim != 5:
            raise ValueError(
                f"Expected fixed/warped shape (B,C,D,H,W), "
                f"got fixed={tuple(fixed.shape)}, warped={tuple(warped.shape)}"
            )

        mu_f = fixed.mean(dim=[2, 3, 4], keepdim=True)
        mu_w = warped.mean(dim=[2, 3, 4], keepdim=True)

        cf = fixed - mu_f
        cw = warped - mu_w

        num = (cf * cw).sum(dim=[2, 3, 4])
        den = torch.sqrt(
            (cf ** 2).sum(dim=[2, 3, 4]) *
            (cw ** 2).sum(dim=[2, 3, 4]) + 1e-8
        )

        return (1.0 - num / den).mean()

    @staticmethod
    def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
        dx = flow[:, 1:, :, :] - flow[:, :-1, :, :]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        dz = flow[:, :, :, 1:] - flow[:, :, :, :-1]

        dx2 = (dx[:, 1:, :, :] - dx[:, :-1, :, :]).pow(2).mean()
        dy2 = (dy[:, :, 1:, :] - dy[:, :, :-1, :]).pow(2).mean()
        dz2 = (dz[:, :, :, 1:] - dz[:, :, :, :-1]).pow(2).mean()

        return dx2 + dy2 + dz2

    def forward(
        self,
        fixeds: Optional[Tuple[torch.Tensor, ...]],
        warpeds: Optional[Tuple[torch.Tensor, ...]],
        disp_fields: Optional[Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        if disp_fields is None:
            if fixeds is not None and len(fixeds) > 0:
                return fixeds[0].new_tensor(0.0)
            return torch.tensor(0.0, dtype=torch.float32)

        if not isinstance(disp_fields, (tuple, list)):
            disp_fields = (disp_fields,)

        ref = disp_fields[0]
        if ref.ndim == 4:
            ref = ref.unsqueeze(0)

        total_ncc = ref.new_tensor(0.0)
        total_smooth = ref.new_tensor(0.0)

        if fixeds is not None and warpeds is not None:
            if len(fixeds) != len(warpeds):
                raise ValueError(f"len(fixeds) != len(warpeds): {len(fixeds)} vs {len(warpeds)}")
            for f, w in zip(fixeds, warpeds):
                total_ncc = total_ncc + self.ncc_loss(f, w)
            total_ncc = total_ncc / len(fixeds)

        for flow in disp_fields:
            total_smooth = total_smooth + self.smoothness_loss(flow)
        total_smooth = total_smooth / len(disp_fields)

        return self.lambda_ncc * total_ncc + self.lambda_smooth * total_smooth


def orthogonal_loss(f_str, f_sig):
    b = f_str.size(0)
    f_str = f_str.flatten(1)
    f_sig = f_sig.flatten(1)

    f_str = F.normalize(f_str, dim=1)
    f_sig = F.normalize(f_sig, dim=1)

    return torch.mean(torch.abs(torch.sum(f_str * f_sig, dim=1)))

class ConcentrationLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-8,
                 temperature: float = 0.1):
        super().__init__()
        self.eps = eps
        self.temperature = temperature

    def forward(self, diff_abs: Optional[torch.Tensor]) -> torch.Tensor:
        if diff_abs is None or diff_abs.numel() == 0:
            return torch.tensor(0.0, device=diff_abs.device if diff_abs is not None else 'cpu')

        B, C, D, H, W = diff_abs.shape
        flat = diff_abs.view(B, C, -1)

        p = F.softmax(flat / self.temperature, dim=-1)

        entropy = - (p * torch.log(p + self.eps)).sum(dim=-1).mean()


        return entropy

class SACANLoss(nn.Module):
    def __init__(
        self,
        config: AblationConfig,
        alpha_bin: float = 1.0,
        alpha_stage: float = 1.0,
        alpha_reg: float = 0.1,
        alpha_decouple: float = 0.05,
        alpha_conc: float = 0.0,
        alpha_local_conc: float = 0.0,
        num_stages: int = 5,
        ordinal: bool = False,
        stage_class_weights: Optional[List[torch.Tensor]] = None,
        bin_class_weights: Optional[List[torch.Tensor]] = None,
        task=None
    ):
        super().__init__()
        self.config = config
        self.alpha_bin = alpha_bin
        self.alpha_stage = alpha_stage
        self.alpha_reg = alpha_reg
        self.alpha_decouple = alpha_decouple
        self.alpha_conc = alpha_conc
        self.alpha_local_conc = alpha_local_conc
        self.task = task

        self.reg_loss = RegistrationLoss()
        self.conc_loss = ConcentrationLoss()

        if self.task == 'stage':
            self.stage_class_weights = stage_class_weights
            self.left_stage_loss = OrdinalStageLoss(num_stages=num_stages, ordinal=ordinal,
                                                    stage_class_weight=self.stage_class_weights[0])
            self.right_stage_loss = OrdinalStageLoss(num_stages=num_stages, ordinal=ordinal,
                                                     stage_class_weight=self.stage_class_weights[1])
        if self.task == 'bin':
            self.bin_class_weights = bin_class_weights
            self.left_bceloss = WeightedBCELoss(pos_weight=self.bin_class_weights[0][1])
            self.right_bceloss = WeightedBCELoss(pos_weight=self.bin_class_weights[1][1])


    def forward(
        self,
        outputs: Dict[str, Optional[torch.Tensor]],
        labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        loss_reg = self.reg_loss(
            fixeds=(outputs.get("xl_t1_aligned"), outputs.get("xl_t2_aligned")),
            warpeds=(outputs.get("xr_t1_aligned"), outputs.get("xr_t2_aligned")),
            disp_fields=outputs.get('disp_fields'))

        loss_conc = self.conc_loss(outputs.get('diff_abs')) \
            if self.config.use_concentration_loss else torch.tensor(0.0, device=device)

        loss_local_conc = torch.tensor(0.0, device=device)
        if (self.alpha_local_conc > 0 and outputs.get('z_left_local') is not None
                and outputs.get('z_right_local') is not None):
            loss_local_conc = (
                self.conc_loss(outputs['z_left_local'].abs()) +
                self.conc_loss(outputs['z_right_local'].abs())
            ) * 0.5

        loss_reg = loss_reg.to(device)
        loss_conc = loss_conc.to(device)
        if outputs['f_left_str'] is not None and outputs['f_left_sig'] is not None:
            loss_str = F.l1_loss(outputs['f_left_str'], outputs['f_right_str'])
            loss_decouple = loss_str + 0.1 * (
                    orthogonal_loss(outputs['f_left_str'], outputs['f_left_sig']) +
                    orthogonal_loss(outputs['f_right_str'], outputs['f_right_sig'])
            )
        else:
            loss_decouple = torch.tensor(0.0, device=device)

        if self.task == 'bin':
            lb = labels['left_binary'].float().unsqueeze(1)
            rb = labels['right_binary'].float().unsqueeze(1)

            loss_bin = self.left_bceloss(outputs['left_binary'], lb) + \
                       self.right_bceloss(outputs['right_binary'], rb)
            total = (self.alpha_bin * loss_bin + self.alpha_reg * loss_reg + self.alpha_decouple * loss_decouple+ self.alpha_local_conc * loss_local_conc)

            return total, \
                        {
                            'total': total.detach(),
                            'binary': loss_bin.detach(),
                            'registration': loss_reg.detach(),
                            'decouple': loss_decouple.detach(),
                            'concentration': loss_conc.detach(),
                        }

        if self.task == 'stage':
            ls = labels['left_stage'].long()
            rs = labels['right_stage'].long()
            loss_stage = self.left_stage_loss(outputs['left_stage'], ls) + \
                     self.right_stage_loss(outputs['right_stage'], rs)

            total = (self.alpha_stage * loss_stage
                     + self.alpha_reg * loss_reg
                     + self.alpha_decouple * loss_decouple
                     + self.alpha_conc * loss_conc
                     + self.alpha_local_conc * loss_local_conc)

            return total, \
                {
                    'total': total.detach(),
                    'stage': loss_stage.detach(),
                    'registration': loss_reg.detach(),
                    'decouple': loss_decouple.detach(),
                    'concentration': loss_conc.detach()
                }


