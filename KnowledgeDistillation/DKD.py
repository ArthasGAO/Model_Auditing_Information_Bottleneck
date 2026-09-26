import torch
import torch.nn as nn
import torch.nn.functional as F

from KnowledgeDistillation.base_distiller import Distiller

# --- DKD Helper Functions ---
def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student + 1e-8) # Added epsilon for numerical stability
    
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        * (temperature**2)
        / target.shape[0]
    )
    
    pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
    
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


# --- DKD Distiller Class ---
class DKD(Distiller):
    """Decoupled Knowledge Distillation"""
    def __init__(self, student, teacher, ce_weight=1.0, alpha=1.0, beta=8.0, temperature=4.0,
                 warmup=20, ce_criterion=None):
        super(DKD, self).__init__(student, teacher)
        # ce_criterion: the ground-truth term. Defaults to plain
        # F.cross_entropy (previous behaviour). Pass the criterion the
        # matching negative models trained with - e.g. timm's
        # LabelSmoothingCrossEntropy(0.1) for the DeiT recipe - so a student
        # and its baseline differ only by the teacher term.
        # NOTE: only the CE term changes. dkd_loss builds its gt / non-gt
        # mask from the hard `target`, which label smoothing does not affect.
        self.ce_criterion = ce_criterion if ce_criterion is not None else F.cross_entropy
        print("DKD metric distillation initializing!")
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup = warmup

    def forward_train(self, image, target, **kwargs):
        # print("DKD metric distillation executing!")
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # kwargs["epoch"] is required for DKD warmup schedule
        current_epoch = kwargs.get("epoch", 0) 

        loss_ce = self.ce_weight * self.ce_criterion(logits_student, target)
        loss_dkd = min(current_epoch / self.warmup, 1.0) * dkd_loss(
            logits_student, logits_teacher, target,
            self.alpha, self.beta, self.temperature
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict