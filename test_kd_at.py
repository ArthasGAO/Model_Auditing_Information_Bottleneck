"""CPU tests for main_kd_at.py (synthetic data, tiny student/teacher, no CIFAR, no GPU).

Run:  python test_kd_at.py
"""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

import at_family_common as common
import main_kd_at as kdat
from KnowledgeDistillation.KD import KD
from Model.kd_eval import KDLogitsOnly
from util import train_one_epoch_kd
from util_adv import NormalizedModel, pgd_attack_v2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class TinyDist(nn.Module):
    """Returns (logits, feats) like ResNet18_dist."""
    def __init__(self, width=4, num_classes=10, with_bn=True):
        super().__init__()
        layers = [nn.Conv2d(3, width, 3, padding=1)]
        if with_bn:
            layers.append(nn.BatchNorm2d(width))
        layers += [nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        f = self.features(x)
        return self.fc(f), [f]


class RecordingTeacher(TinyDist):
    def __init__(self):
        super().__init__(width=6)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return super().forward(x)


def synthetic(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 8, 8, generator=g), torch.randint(0, 10, (n,), generator=g)


def normalise(x):
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def make_distiller(seed, with_bn=True):
    torch.manual_seed(seed)
    teacher = TinyDist(width=6, with_bn=with_bn)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    student = TinyDist(width=4, with_bn=with_bn)
    return KD(student, teacher, temperature=4.0, ce_weight=0.1, kd_weight=0.9)


class KDATTests(unittest.TestCase):

    def test_lambda_zero_equals_train_one_epoch_kd(self):
        """lambda = 0 (bn_policy both): the new loop must reproduce util.train_one_epoch_kd."""
        x, y = synthetic()
        old = make_distiller(1)
        new = copy.deepcopy(old)
        old_loader = DataLoader(TensorDataset(normalise(x), y), batch_size=16, shuffle=False)
        new_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
        opt_old = torch.optim.SGD(old.get_learnable_parameters(), lr=0.1)
        opt_new = torch.optim.SGD(new.get_learnable_parameters(), lr=0.1)
        student_w = NormalizedModel(KDLogitsOnly(new.student), MEAN, STD)

        def never(model, xx, t):
            raise AssertionError("no attack must be generated when lambda == 0")

        r_old = train_one_epoch_kd(old, old_loader, opt_old, 0, 'cpu')
        r_new = kdat.kd_at_one_epoch(new, student_w, kdat.Normalizer(MEAN, STD), new_loader, opt_new, never,
                                     'cpu', clean_weight=1.0, adv_weight=0.0, bn_policy="both")
        self.assertLess(abs(r_old["train_loss"] - r_new["train_loss"]), 1e-5)
        self.assertLess(abs(r_old["train_acc"] - r_new["train_acc"]), 1e-9)
        for (n1, p1), (n2, p2) in zip(old.student.state_dict().items(), new.student.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1.float(), p2.float(), atol=1e-5), n1)

    def test_teacher_sees_only_the_clean_batch(self):
        x, y = synthetic(n=16)
        torch.manual_seed(2)
        teacher = RecordingTeacher()
        for p in teacher.parameters():
            p.requires_grad = False
        distiller = KD(TinyDist(width=4), teacher, temperature=4.0, ce_weight=0.1, kd_weight=0.9)
        student_w = NormalizedModel(KDLogitsOnly(distiller.student), MEAN, STD)
        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
        opt = torch.optim.SGD(distiller.get_learnable_parameters(), lr=0.0)
        gen = common.make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 3})
        kdat.kd_at_one_epoch(distiller, student_w, kdat.Normalizer(MEAN, STD), loader, opt, gen, 'cpu',
                             clean_weight=1.0, adv_weight=1.0, bn_policy="both")
        self.assertEqual(len(teacher.inputs), 1)                       # once per batch
        self.assertTrue(torch.allclose(teacher.inputs[0], normalise(x), atol=1e-6))   # the clean batch
        self.assertFalse(teacher.training)

    def test_additive_loss_is_kd_plus_lambda_kd_unscaled(self):
        x, y = synthetic(n=16)
        distiller = make_distiller(3, with_bn=False)
        student_w = NormalizedModel(KDLogitsOnly(distiller.student), MEAN, STD)
        with torch.no_grad():
            t_logits, _ = distiller.teacher(normalise(x))
        loss_fn = kdat.kd_batch_loss_fn(distiller, t_logits)
        identity = lambda m, xx, t: xx
        for lam in (1.0, 2.5):
            out = common.mixed_adversarial_step(student_w, x, y, loss_fn, identity, 1.0, lam, bn_policy="both")
            self.assertTrue(torch.allclose(out["loss"], (1.0 + lam) * out["clean_loss"], atol=1e-6))
        # and the clean term is KD.forward_train's objective
        distiller.train(); distiller.teacher.eval()
        _, losses = distiller.forward_train(normalise(x), y)
        self.assertTrue(torch.allclose(sum(losses.values()), out["clean_loss"], atol=1e-6))

    def test_names_and_plan_expansion(self):
        plan = {
            "Scenario_Name": "CIFAR-10_ResNet-34to18_25000",
            "Distillation": [{"name": "KD", "params": {}}, {"name": "DKD", "params": {}}],
            "AdversarialTraining": {"form": "additive", "lambda": 1.0, "bn_policy": "both"},
            "Attack": {"name": "PGD", "eps": [0.007843, 0.031373], "steps": 10},
            "Epochs": 200,
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plan.yaml"
            p.write_text(yaml.safe_dump(plan), encoding="utf-8")
            _, names = kdat.expand_plan(p, seeds=[0, 1], rates=[0.0], methods=["KD"], run_tag="v1")
            self.assertEqual(len(names), 4)
            self.assertEqual(names[1], "CIFAR-10_ResNet-34to18_25000_KD_0_0.0"
                                       "_ATPGD_eps=0.031373_steps=10_lambda=1_bn=both_run=v1")
            with self.assertRaises(NotImplementedError):
                kdat.expand_plan(p, seeds=[0], rates=[0.0], methods=["DKD"], run_tag="v1")
            with self.assertRaises(ValueError):
                kdat.expand_plan(p, seeds=[0], rates=[0.0], methods=["KD"], run_tag="v1", eps_filter=[0.05])


if __name__ == "__main__":
    unittest.main()
