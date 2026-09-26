"""CPU tests for at_family_common.py and main_ft_at.py (synthetic data, no CIFAR, no GPU).

Run:  python test_ft_at.py
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
import main_ft_at as fat
from util import ft_one_epoch, setup_finetune
from util_adv import NormalizedModel, pgd_attack_v2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class TinyResNetLike(nn.Module):
    """conv-bn backbone + .fc head, so setup_finetune's ResNet branch applies."""
    def __init__(self, num_classes=10, with_bn=True):
        super().__init__()
        layers = [nn.Conv2d(3, 4, 3, padding=1)]
        if with_bn:
            layers.append(nn.BatchNorm2d(4))
        layers += [nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.features(x))


def synthetic(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 8, 8, generator=g), torch.randint(0, 10, (n,), generator=g)


def normalise(x):
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def loaders(x, y):
    return (DataLoader(TensorDataset(normalise(x), y), batch_size=16, shuffle=False),
            DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False))


def first_bn(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)][0]


class FTATTests(unittest.TestCase):

    def test_lambda_zero_equals_ft_one_epoch(self):
        """lambda = 0: the new loop must reproduce util.ft_one_epoch (FT-AL) exactly."""
        x, y = synthetic()
        torch.manual_seed(1)
        base_old = TinyResNetLike()
        base_new = copy.deepcopy(base_old)
        model = NormalizedModel(base_new, MEAN, STD)
        old_loader, new_loader = loaders(x, y)
        crit = nn.CrossEntropyLoss()
        opt_old = torch.optim.SGD(base_old.parameters(), lr=0.1)
        opt_new = torch.optim.SGD(base_new.parameters(), lr=0.1)

        def never(model, x, t):
            raise AssertionError("no attack must be generated when lambda == 0")

        # bn_policy="both": with no adversarial term this is the plain train-mode
        # loop. (Under the default "frozen" policy, lambda=0 is FT with frozen BN.)
        old = ft_one_epoch(base_old, old_loader, opt_old, crit, 0, 'cpu', "FT-AL")
        new = fat.ft_at_one_epoch(model, base_new, new_loader, opt_new, crit, never, "FT-AL",
                                  clean_weight=1.0, adv_weight=0.0, device='cpu', bn_policy="both")
        self.assertLess(abs(old["train_loss"] - new["train_loss"]), 1e-5)
        self.assertLess(abs(old["train_acc"] - new["train_acc"]), 1e-9)
        for (n1, p1), (n2, p2) in zip(base_old.state_dict().items(), base_new.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1.float(), p2.float(), atol=1e-5), n1)

    def test_additive_loss_is_clean_plus_lambda_adv_unscaled(self):
        """With x_adv := x on a BN-free net, L must equal (1 + lambda) * CE(x): no normalisation."""
        x, y = synthetic(n=16)
        torch.manual_seed(4)
        model = NormalizedModel(TinyResNetLike(with_bn=False), MEAN, STD)
        crit = nn.CrossEntropyLoss()
        identity = lambda m, xx, t: xx
        for lam in (1.0, 2.5):
            out = common.mixed_adversarial_step(model, x, y, crit, identity, 1.0, lam)
            self.assertTrue(torch.allclose(out["loss"], (1.0 + lam) * out["clean_loss"], atol=1e-6))
            self.assertTrue(torch.allclose(out["adv_loss"], out["clean_loss"], atol=1e-6))

    def _bn_after_policy(self, policy, seed=2):
        x, y = synthetic(n=16, seed=seed)
        torch.manual_seed(seed)
        base = TinyResNetLike()
        reference = copy.deepcopy(base)
        model = NormalizedModel(base, MEAN, STD)
        gen = common.make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 3})
        out = common.mixed_adversarial_step(model, x, y, nn.CrossEntropyLoss(), gen, 1.0, 1.0,
                                            bn_policy=policy)
        self.assertTrue(model.training)          # leaves the model in train mode
        self.assertIsNotNone(out["adv_logits"])
        reference.train()
        reference(normalise(x))                  # one clean train-mode forward
        return first_bn(base), first_bn(reference)

    def test_bn_policy_both_updates_bn_twice(self):
        bn, ref = self._bn_after_policy("both")
        self.assertEqual(bn.num_batches_tracked.item(), 2)
        self.assertFalse(torch.allclose(bn.running_mean, ref.running_mean, atol=1e-6))

    def test_bn_policy_frozen_never_touches_running_stats(self):
        x, y = synthetic(n=16, seed=5)
        torch.manual_seed(5)
        base = TinyResNetLike()
        before = {k: v.clone() for k, v in base.state_dict().items() if "running" in k or "num_batches" in k}
        fc_before = base.fc.weight.detach().clone()
        bn_weight_before = first_bn(base).weight.detach().clone()
        model = NormalizedModel(base, MEAN, STD)
        opt = torch.optim.SGD(base.parameters(), lr=0.1)
        gen = common.make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 3})
        out = common.mixed_adversarial_step(model, x, y, nn.CrossEntropyLoss(), gen, 1.0, 1.0, bn_policy="frozen")
        out["loss"].backward()
        opt.step()
        after = base.state_dict()
        for k, v in before.items():
            self.assertTrue(torch.equal(v, after[k]), k)     # running stats and counter untouched
        self.assertIsNotNone(out["clean_logits"])
        self.assertIsNotNone(out["adv_logits"])
        self.assertFalse(first_bn(base).training)           # BN layers left in eval
        self.assertTrue(model.training)                      # the module tree itself is in train mode
        self.assertFalse(torch.equal(fc_before, after["fc.weight"]))            # weights trained
        self.assertFalse(torch.equal(bn_weight_before, first_bn(base).weight))  # BN affine trained

    def test_bn_policy_clean_and_clean_eval_update_bn_from_clean_only(self):
        for policy in ("clean", "clean_eval"):
            bn, ref = self._bn_after_policy(policy)
            self.assertEqual(bn.num_batches_tracked.item(), 1, policy)
            self.assertTrue(torch.allclose(bn.running_mean, ref.running_mean, atol=1e-6), policy)
            self.assertTrue(torch.allclose(bn.running_var, ref.running_var, atol=1e-6), policy)
        with self.assertRaises(ValueError):
            self._bn_after_policy("nope")

    def test_ft_ll_only_head_changes_and_backbone_bn_frozen(self):
        x, y = synthetic(n=32)
        torch.manual_seed(3)
        base = TinyResNetLike()
        before = copy.deepcopy(base.state_dict())
        base = setup_finetune(model=base, strategy="FT-LL", device='cpu')
        model = NormalizedModel(base, MEAN, STD)
        _, loader = loaders(x, y)
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, base.parameters()), lr=0.1)
        gen = common.make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 3})
        fat.ft_at_one_epoch(model, base, loader, opt, nn.CrossEntropyLoss(), gen, "FT-LL",
                            clean_weight=1.0, adv_weight=1.0, device='cpu')
        after = base.state_dict()
        for k in before:
            if k.startswith("features."):
                self.assertTrue(torch.equal(before[k], after[k]), f"backbone changed: {k}")
        self.assertFalse(torch.equal(before["fc.weight"], after["fc.weight"]))

    def test_config_and_names(self):
        plan = {
            "Scenario_Name": "CIFAR-10_ResNet-18_25000_Same_25000",
            "FT_Dataset": {"group_size": 25000},
            "Optimizers": [{"strategy": "FT-LL"}, {"strategy": "FT-AL"}, {"strategy": "RT-AL"}],
            "AdversarialTraining": {"form": "additive", "lambda": 1.0},
            "Attack": {"name": "PGD", "eps": 0.031373, "steps": 10},
        }
        attacks, cw, aw, bn = common.parse_additive_at_config(plan)
        self.assertEqual((cw, aw, len(attacks), bn), (1.0, 1.0, 1, "frozen"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plan.yaml"
            p.write_text(yaml.safe_dump(plan), encoding="utf-8")
            _, names = fat.expand_plan(p, seeds=[0, 1], strategies=["FT-AL", "RT-AL"], run_tag="v1")
            self.assertEqual(len(names), 4)
            self.assertEqual(names[0], "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_FT-AL_ftsize=25000_ftseed=0"
                                       "_ATPGD_eps=0.031373_steps=10_lambda=1_bn=frozen_run=v1")
            with self.assertRaises(ValueError):
                fat.expand_plan(p, seeds=[0], strategies=["FT-XX"], run_tag="v1")
        _, _, _, bn2 = common.parse_additive_at_config(
            {**plan, "AdversarialTraining": {"lambda": 1.0, "bn_policy": "clean_eval"}})
        self.assertEqual(bn2, "clean_eval")
        self.assertTrue(common.at_suffix("PGD", {"eps": 0.1}, 1.0, "v1", "clean_eval").endswith("_bn=clean_eval_run=v1"))
        with self.assertRaises(ValueError):
            common.parse_additive_at_config({**plan, "AdversarialTraining": {"lambda": -1}})
        with self.assertRaises(ValueError):
            common.parse_additive_at_config({**plan, "AdversarialTraining": {"form": "mixed"}})
        with self.assertRaises(ValueError):
            common.parse_additive_at_config({**plan, "AdversarialTraining": {"bn_policy": "adv"}})
        with self.assertRaises(ValueError):
            common.at_suffix("PGD", {"eps": 0.1}, 1.0, "bad tag")


if __name__ == "__main__":
    unittest.main()
