"""CPU tests for main_prune_at.py (synthetic data, tiny pruned net, no CIFAR, no GPU).

Run:  python test_prune_at.py
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
import main_prune_at as pat
from util import ft_one_epoch, prune_model_global, remove_prune_mask
from util_adv import NormalizedModel, pgd_attack_v2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class TinyResNetLike(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc(self.features(x))


def synthetic(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 8, 8, generator=g), torch.randint(0, 10, (n,), generator=g)


def normalise(x):
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def pruned(seed, amount=0.5):
    torch.manual_seed(seed)
    net = TinyResNetLike()
    return prune_model_global(net, amount=amount)


def masks(net):
    return {n: m.weight_mask.clone() for n, m in net.named_modules() if hasattr(m, "weight_mask")}


class PruneATTests(unittest.TestCase):

    def test_lambda_zero_equals_ft_one_epoch_on_pruned_net(self):
        x, y = synthetic()
        old = pruned(1)
        pat.refresh_pruned_weights(old)
        new = copy.deepcopy(old)
        model = NormalizedModel(new, MEAN, STD)
        old_loader = DataLoader(TensorDataset(normalise(x), y), batch_size=16, shuffle=False)
        new_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
        crit = nn.CrossEntropyLoss()
        opt_old = torch.optim.SGD(filter(lambda p: p.requires_grad, old.parameters()), lr=0.1)
        opt_new = torch.optim.SGD(filter(lambda p: p.requires_grad, new.parameters()), lr=0.1)

        def never(m, xx, t):
            raise AssertionError("no attack must be generated when lambda == 0")

        r_old = ft_one_epoch(old, old_loader, opt_old, crit, 0, 'cpu', "FT-AL")
        r_new = fat.ft_at_one_epoch(model, new, new_loader, opt_new, crit, never, "FT-AL",
                                    clean_weight=1.0, adv_weight=0.0, device='cpu', bn_policy="both")
        self.assertLess(abs(r_old["train_loss"] - r_new["train_loss"]), 1e-5)
        pat.refresh_pruned_weights(old); pat.refresh_pruned_weights(new)
        for (n1, p1), (n2, p2) in zip(old.state_dict().items(), new.state_dict().items()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1.float(), p2.float(), atol=1e-5), n1)

    def test_masks_and_sparsity_survive_an_adversarial_epoch_and_dense_save(self):
        x, y = synthetic(n=32)
        net = pruned(2, amount=0.5)
        before = masks(net)
        sp_before = pat.global_sparsity(net)
        model = NormalizedModel(net, MEAN, STD)
        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
        gen = common.make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 3})
        fat.ft_at_one_epoch(model, net, loader, opt, nn.CrossEntropyLoss(), gen, "FT-AL",
                            clean_weight=1.0, adv_weight=1.0, device='cpu', bn_policy="frozen")
        after = masks(net)
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]), k)
        self.assertAlmostEqual(pat.global_sparsity(net), sp_before, places=6)
        # dense_state_dict works right after a grad-enabled forward and contains real zeros
        state = pat.dense_state_dict(net)
        self.assertNotIn("features.0.weight_mask", state)
        self.assertIn("features.0.weight", state)
        w = state["features.0.weight"]
        self.assertAlmostEqual((w == 0).float().mean().item(), 0.5, delta=0.15)
        self.assertTrue(hasattr(net.features[0], "weight_mask"))   # the live model keeps its masks

    def test_frozen_keeps_bn_stats_and_recalibration_changes_them(self):
        x, y = synthetic(n=32)
        net = pruned(3)
        model = NormalizedModel(net, MEAN, STD)
        bn = net.features[1]
        rm0 = bn.running_mean.clone()
        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
        n = pat.recalibrate_bn(model, loader, 'cpu')
        self.assertEqual(n, 2)
        self.assertFalse(torch.allclose(bn.running_mean, rm0))
        self.assertEqual(bn.momentum, 0.1)
        self.assertFalse(model.training)
        rm1 = bn.running_mean.clone()
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
        gen = common.make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 3})
        fat.ft_at_one_epoch(model, net, loader, opt, nn.CrossEntropyLoss(), gen, "FT-AL",
                            clean_weight=1.0, adv_weight=1.0, device='cpu', bn_policy="frozen")
        self.assertTrue(torch.equal(bn.running_mean, rm1))        # frozen after recalibration

    def test_names_and_plan_expansion(self):
        plan = {
            "Scenario_Name": "CIFAR-10_ResNet-18_25000_Same_25000",
            "FT_Dataset": {"group_size": 25000},
            "Optimizers": [{"sparsity": 0.2}, {"sparsity": 0.8}],
            "AdversarialTraining": {"form": "additive", "lambda": 1.0, "bn_policy": "frozen",
                                    "recalibrate_bn_after_prune": True},
            "Attack": {"name": "PGD", "eps": [0.007843, 0.031373], "steps": 10},
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plan.yaml"
            p.write_text(yaml.safe_dump(plan), encoding="utf-8")
            _, names = pat.expand_plan(p, seeds=[0], sparsities=[0.8], run_tag="v1")
            self.assertEqual(len(names), 2)
            self.assertEqual(names[1], "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_sparsity=0.8_FT-AL_ftsize=25000"
                                       "_ftseed=0_ATPGD_eps=0.031373_steps=10_lambda=1_bn=frozen_recal=clean_run=v1")
            with self.assertRaises(ValueError):
                pat.expand_plan(p, seeds=[0], sparsities=[0.5], run_tag="v1")
            plan["AdversarialTraining"]["recalibrate_bn_after_prune"] = False
            p.write_text(yaml.safe_dump(plan), encoding="utf-8")
            _, names = pat.expand_plan(p, seeds=[0], sparsities=[0.8], run_tag="v1", eps_filter=[0.031373])
            self.assertEqual(len(names), 1)
            self.assertIn("_recal=none_", names[0])
        with self.assertRaises(ValueError):
            pat.parse_recal({"AdversarialTraining": {"recalibrate_bn_after_prune": "yes"}})


if __name__ == "__main__":
    unittest.main()
