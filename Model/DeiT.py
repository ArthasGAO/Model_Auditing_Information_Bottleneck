import torch
import torch.nn as nn


def split_deit_outputs(outputs):
    """
    timm DeiT distilled models may return:
      - tuple/list: (cls_logits, dist_logits)
      - tensor: logits (already merged or non-distilled model)
    """
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        #print("Distillation Logits Received!")
        return outputs[0], outputs[1]
    return outputs, None


class DistillationLoss(nn.Module): # this represents the overall training loss used for target model training
    """
    Supports:
      - no distillation (enabled=False) -> acts like base criterion on cls head
      - hard distillation (teacher argmax)
      - soft distillation (KL with temperature)
    """
    def __init__(self, teacher, base_criterion, distill_type="hard", alpha=0.5, tau=2.0):
        super().__init__()
        self.teacher = teacher
        self.base_criterion = base_criterion
        assert distill_type in ['none', 'soft', 'hard']
        self.distill_type = distill_type
        self.alpha = float(alpha)
        self.tau = float(tau)

        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()


    def forward(self, inputs, outputs, targets):
        cls_logits, dist_logits = split_deit_outputs(outputs)

        #print(f"cls_logtis is {cls_logits}, dist_logits is {dist_logits} !")

        # Base loss always on cls head
        base_loss = self.base_criterion(cls_logits, targets)

        # If distillation off OR no distillation head exists -> return base
        if (self.distill_type == 'none') or (self.teacher is None) or (dist_logits is None):
            return base_loss

        with torch.no_grad():
            t_logits = self.teacher(inputs)

        if self.distill_type.lower() == "hard":
            t_labels = torch.argmax(t_logits, dim=1)
            dist_loss = nn.functional.cross_entropy(dist_logits, t_labels)

        elif self.distill_type.lower() == "soft":
            # KL(student || teacher) with temperature
            T = self.tau
            t_probs = torch.softmax(t_logits / T, dim=1)
            s_log_probs = torch.log_softmax(dist_logits / T, dim=1)
            dist_loss = nn.functional.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

        else:
            raise ValueError(f"Unknown distill_type: {self.distill_type}")

        #print(base_loss)
        #print(dist_loss)
        return (1.0 - self.alpha) * base_loss + self.alpha * dist_loss
    

class FusionCELoss(nn.Module): # this represents the way in FT to also output the distillation logits but
                                # use the mix of dist and cls to update the model simultaneously
    def __init__(self, base_criterion, alpha):
        super().__init__()
        self.base = base_criterion
        self.alpha = alpha

    def forward(self, outputs, targets):
        cls_logits, dist_logits = split_deit_outputs(outputs)
        if dist_logits is None:
            return self.base(cls_logits, targets)
        fused = (1.0 - self.alpha) * cls_logits + self.alpha * dist_logits
        return self.base(fused, targets)


class DeiTKDStudent(nn.Module):
    """
    Adapt a timm DeiT backbone to the KnowledgeDistillation `(logits, feats)`
    contract used by KnowledgeDistillation.base_distiller.Distiller.

    Logit-based distillers (KD, DKD) read only the first return value, so they
    work unchanged. The `feats` dict is returned for interface compatibility but
    carries an empty spatial list: a ViT produces token sequences [B, N, D], not
    conv feature maps, so there is no layer that FitNet's ConvReg hint can align
    against. FitNet raises a clear error on this model rather than failing deep
    inside the MSE (see KnowledgeDistillation/FitNet.py).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @property
    def num_classes(self):
        # util.collect_logits infers the class count from the model, probing
        # .fc / .classifier / .num_classes. The wrapper exposes none of those,
        # so forward the timm backbone's attribute.
        return self.backbone.num_classes

    def forward(self, x):
        cls_logits, _ = split_deit_outputs(self.backbone(x))
        feats = {"feats": [], "preact_feats": [], "pooled_feat": None}
        return cls_logits, feats
