"""Exact KD student architectures, exposing logits for ownership evaluation."""
import torch.nn as nn


def build_kd_student(model_name, num_classes, model_config=None):
    if model_name == 'ResNet-18':
        from Model.ResNet_18_dist import ResNet18_dist
        return ResNet18_dist(num_classes=num_classes)
    if model_name == 'VGG16':
        from Model.VGG16_dist import VGG16_Teacher
        return VGG16_Teacher(num_classes=num_classes)
    if model_name == 'DeiT':
        from Model.DeiT import DeiTKDStudent
        from util import build_deit_student
        if not model_config:
            raise ValueError('KD DeiT requires Model_Config from its Student_Model training configuration.')
        return DeiTKDStudent(build_deit_student({'Model': {**model_config, 'pretrained': False}}, num_classes))
    raise ValueError(f'Unsupported KD student architecture: {model_name}')


class KDLogitsOnly(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, x):
        logits, _ = self.student(x)
        return logits
