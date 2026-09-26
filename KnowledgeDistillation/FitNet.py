# --- FitNet Helper Class ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from KnowledgeDistillation.base_distiller import Distiller


class ConvReg(nn.Module):
    """1x1 Convolution to align Student channel dimensions to Teacher channel dimensions."""
    def __init__(self, s_channels, t_channels):
        super(ConvReg, self).__init__()
        self.conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(t_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


# --- FitNet Distiller Class ---
class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""
    def __init__(self, student, teacher, ce_weight=1.0, feat_weight=100.0, hint_layer=2,
                 input_size=(32, 32), ce_criterion=None):
        super(FitNet, self).__init__(student, teacher)
        # ce_criterion: the ground-truth term. Defaults to plain
        # F.cross_entropy (previous behaviour). Pass the criterion the
        # matching negative models trained with - e.g. timm's
        # LabelSmoothingCrossEntropy(0.1) for the DeiT recipe - so a student
        # and its baseline differ only by the teacher term.
        self.ce_criterion = ce_criterion if ce_criterion is not None else F.cross_entropy
        print("FitNet metric distillation initializing!")
        self.ce_weight = ce_weight
        self.feat_weight = feat_weight
        self.hint_layer = hint_layer

        # Dynamically determine the channel sizes using a dummy tensor
        # This replaces the need for the external get_feat_shapes function

        self.student.eval()

        device = next(teacher.parameters()).device
        dummy_img = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        with torch.no_grad():
            _, feat_s = self.student(dummy_img)
            _, feat_t = self.teacher(dummy_img)

        # --- Architecture compatibility check -------------------------------
        # ConvReg is a 1x1 convolution: it aligns CHANNELS only, never spatial
        # resolution. Hint maps must therefore already share H x W. This holds
        # within a family (ResNet-18/ResNet-10 both give (128,16,16) at index 2;
        # VGG16/VGG8 both give (256,4,4)) but not across families, and a token
        # based student (DeiT) exposes no spatial maps at all.
        for tag, feat in (("student", feat_s), ("teacher", feat_t)):
            if len(feat["feats"]) <= self.hint_layer:
                raise ValueError(
                    f"FitNet: {tag} exposes {len(feat['feats'])} feature stage(s), "
                    f"so hint_layer={self.hint_layer} is out of range. A model without "
                    f"spatial feature maps (e.g. DeiT) cannot be used with FitNet; "
                    f"use a logit-based method (KD / DKD) instead."
                )

        s_shape = tuple(feat_s["feats"][self.hint_layer].shape[2:])
        t_shape = tuple(feat_t["feats"][self.hint_layer].shape[2:])
        if s_shape != t_shape:
            raise ValueError(
                f"FitNet: hint_layer={self.hint_layer} has spatial size {s_shape} on the "
                f"student but {t_shape} on the teacher. ConvReg aligns channels only, so "
                f"the MSE would fail. Use a matching hint stage for this architecture "
                f"pair, or use a logit-based method (KD / DKD)."
            )
        # --------------------------------------------------------------------

        s_channels = feat_s["feats"][self.hint_layer].shape[1]
        t_channels = feat_t["feats"][self.hint_layer].shape[1]

        self.conv_reg = ConvReg(s_channels, t_channels).to(device)

    def get_learnable_parameters(self):
        # Must include the conv_reg parameters so the optimizer updates them!
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = sum(p.numel() for p in self.conv_reg.parameters())
        return num_p

    def forward_train(self, image, target, **kwargs):
        # print("FitNet metric distillation executing!")
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        loss_ce = self.ce_weight * self.ce_criterion(logits_student, target)
        
        # Project student features to match teacher features
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        
        loss_feat = self.feat_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict