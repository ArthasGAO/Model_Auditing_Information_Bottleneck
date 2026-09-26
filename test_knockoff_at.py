"""CPU tests for main_knockoff_extraction_at.py (synthetic data, no CIFAR, no GPU).

Run:  python test_knockoff_at.py
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
import main_knockoff_extraction_at as kat
from util import train_one_epoch_knockoff
from util_adv import NormalizedModel, pgd_attack_v2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def tiny_net(num_classes=10, with_bn=True):
    layers = [nn.Conv2d(3, 4, 3, padding=1)]
    if with_bn:
        layers.append(nn.BatchNorm2d(4))
    layers += [nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, num_classes)]
    return nn.Sequential(*layers)


def synthetic(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_raw = torch.rand(n, 3, 8, 8, generator=g)
    soft = torch.softmax(torch.randn(n, 10, generator=g) * 3, dim=1)
    return x_raw, soft


def normalise(x_raw):
    return (x_raw - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


class KnockoffATTests(unittest.TestCase):

    def test_lambda_zero_equals_plain_knockoff_epoch(self):
        """lambda = 0 (bn_policy both): the new loop must reproduce util.train_one_epoch_knockoff."""
        x_raw, soft = synthetic()
        torch.manual_seed(1)
        base_old = tiny_net()
        base_new = copy.deepcopy(base_old)
        new_model = NormalizedModel(base_new, MEAN, STD)
        old_loader = DataLoader(TensorDataset(normalise(x_raw), soft), batch_size=16, shuffle=False)
        new_loader = DataLoader(TensorDataset(x_raw, soft), batch_size=16, shuffle=False)
        opt_old = torch.optim.SGD(base_old.parameters(), lr=0.1)
        opt_new = torch.optim.SGD(new_model.parameters(), lr=0.1)

        def never(model, x, s):
            raise AssertionError("no attack must be generated when lambda == 0")

        old = train_one_epoch_knockoff(base_old, old_loader, opt_old, kat.soft_label_loss, 0, 'cpu')
        new = kat.knockoff_at_one_epoch(new_model, new_loader, opt_new, never, 'cpu',
                                        clean_weight=1.0, adv_weight=0.0, bn_policy="both")
        self.assertLess(abs(old["train_loss"] - new["train_loss"]), 1e-5)
        self.assertLess(abs(old["train_acc"] - new["train_acc"]), 1e-9)
        for (n1, p1), (n2, p2) in zip(base_old.state_dict().items(), base_new.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1.float(), p2.float(), atol=1e-5), n1)

    def test_additive_loss_is_kl_plus_lambda_kl_unscaled(self):
        x_raw, soft = synthetic(n=16)
        torch.manual_seed(4)
        model = NormalizedModel(tiny_net(with_bn=False), MEAN, STD)
        identity = lambda m, xx, t: xx
        for lam in (1.0, 2.5):
            out = common.mixed_adversarial_step(model, x_raw, soft, kat.soft_label_loss, identity, 1.0, lam,
                                                bn_policy="both")
            self.assertTrue(torch.allclose(out["loss"], (1.0 + lam) * out["clean_loss"], atol=1e-6))

    def test_hard_generator_uses_argmax_and_soft_generator_uses_soft_labels(self):
        x_raw, soft = synthetic(n=8)
        seen = {}

        def stub_attack(model, x, y, eps, steps):
            seen["y"] = y.clone()
            return x

        gen = kat.make_knockoff_adv_generator(stub_attack, "PGD", {"eps": 0.01, "steps": 2}, "hard")
        gen(None, x_raw, soft)
        self.assertTrue(torch.equal(seen["y"], soft.argmax(1)))
        model = NormalizedModel(tiny_net(), MEAN, STD).eval()
        gen_soft = kat.make_knockoff_adv_generator(pgd_attack_v2, "PGD", {"eps": 8 / 255, "steps": 3}, "soft")
        x_adv = gen_soft(model, x_raw, soft)
        self.assertLessEqual((x_adv - x_raw).abs().max().item(), 8 / 255 + 1e-6)
        self.assertGreaterEqual(x_adv.min().item(), 0.0)
        self.assertLessEqual(x_adv.max().item(), 1.0)
        self.assertFalse(torch.allclose(x_adv, x_raw))
        with self.assertRaises(ValueError):
            kat.make_knockoff_adv_generator(pgd_attack_v2, "CW", {}, "soft")

    def test_both_policy_updates_bn_from_both_forwards(self):
        x_raw, soft = synthetic(n=16)
        torch.manual_seed(2)
        base = tiny_net()
        model = NormalizedModel(base, MEAN, STD)
        gen = kat.make_knockoff_adv_generator(pgd_attack_v2, "PGD", {"eps": 8 / 255, "steps": 3}, "hard")
        common.mixed_adversarial_step(model, x_raw, soft, kat.soft_label_loss, gen, 1.0, 1.0, bn_policy="both")
        bn = [m for m in base.modules() if isinstance(m, nn.BatchNorm2d)][0]
        self.assertEqual(bn.num_batches_tracked.item(), 2)

    def test_scenario_names_plan_expansion_and_eps_filter(self):
        plan = {
            "Scenario_Name": "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18",
            "AdversarialTraining": {"form": "additive", "lambda": 1.0, "bn_policy": "both", "inner_target": "hard"},
            "Attack": {"name": "PGD", "eps": [0.007843, 0.031373], "steps": 10},
            "Epochs": 160,
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plan.yaml"
            p.write_text(yaml.safe_dump(plan), encoding="utf-8")
            _, names = kat.expand_plan(p, seeds=[0, 1], run_tag="v1")
            self.assertEqual(len(names), 4)
            self.assertEqual(names[1], "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18_0_1.0"
                                       "_ATPGD_eps=0.031373_steps=10_lambda=1_bn=both_inner=hard_run=v1")
            _, only8 = kat.expand_plan(p, seeds=[0], run_tag="v1", eps_filter=[0.031373])
            self.assertEqual(len(only8), 1)
            with self.assertRaises(ValueError):
                kat.expand_plan(p, seeds=[0], run_tag="v1", eps_filter=[0.05])
        with self.assertRaises(ValueError):
            kat.parse_inner_target({"AdversarialTraining": {"inner_target": "argmax"}})


if __name__ == "__main__":
    unittest.main()
