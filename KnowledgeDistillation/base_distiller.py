import torch
import torch.nn as nn
import torch.nn.functional as F

def dbg(tag, msg):
    print(f"[DEBUG][{tag}] {msg}")

# ==========================================
# 1. Base Distiller Class
# ==========================================
class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        dbg("MODEL", f"Teacher model built: {self.teacher.__class__.__name__}")
        dbg("MODEL", f"Student built: {self.student.__class__.__name__}")

    def train(self, mode=True):
        """
        Crucial Override: Ensures the teacher ALWAYS stays in eval mode,
        even when you call distiller.train() in your main loop.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval() # Force teacher to eval
        return self

    def get_learnable_parameters(self):
        return [p for p in self.student.parameters() if p.requires_grad]
        # return [v for k, v in self.student.named_parameters()]

    def forward_train(self, **kwargs):
        raise NotImplementedError()

    def forward_test(self, image):
        # Only use the student for testing/inference
        out = self.student(image)
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])