"""CPU tests for main_at_posthoc.py (unittest; run: python -m unittest test_at_posthoc -v)."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import main_at_posthoc as mp
from at_family_common import make_adv_generator
from main_ft_at import ft_at_one_epoch
from Model.ResNet_18 import ResNet18
from util_adv import NormalizedModel, pgd_attack_v2

PLAN = {
    "Model_Path": ["/ft_final/A_42_1.0_FT-AL_ftsize=25000_ftseed=0", "/kd_final/B_DKD_0_0.0",
                   "/extraction_final/C_Knockoff_Same100_Same18_0_1.0"],
    "Dataset": {"name": "CIFAR-100", "group_size": 25000},
    "AdversarialTraining": {"form": "additive", "lambda": 1.0, "bn_policy": "frozen"},
    "Attack": {"name": "PGD", "eps": [0.007843, 0.015686, 0.031373], "steps": 10},
    "Optimizer": {"name": "SGD", "params": {"lr": 0.001}, "Epochs": 30},
}


class TestNamesAndPlan(unittest.TestCase):
    def test_name_fields_and_validation(self):
        name = mp.posthoc_scenario_name("/kd_final/B_DKD_0_0.0", "PGD", {"eps": 0.031373, "steps": 10},
                                        bn_policy="frozen", train_size=25000, epochs=30, lr=0.001, lam=1.0,
                                        run_tag="v1", at_seed=0)
        self.assertEqual(name, "B_DKD_0_0.0_PGD_eps=0.031373_steps=10_bn=frozen_atn=25000_atepochs=30"
                               "_lr=0.001_lambda=1_advfrac=1_run=v1_atseed=0_mix=add")
        with self.assertRaises(ValueError):
            mp.posthoc_scenario_name("/x", "PGD", {}, bn_policy="frozen", train_size=1, epochs=1, lr=0.1,
                                     lam=1, run_tag="bad tag", at_seed=0)

    @unittest.skipUnless(os.name == "nt", "MAX_PATH check only applies on Windows")
    def test_windows_path_length_check(self):
        mp.check_path_lengths(["short_name"])
        with self.assertRaises(OSError):
            mp.check_path_lengths(["x" * 300])

    def test_expand_plan_filters_and_overrides(self):
        _, runs = mp.expand_plan(PLAN, [0], "v1")
        self.assertEqual(len(runs), 9)
        self.assertEqual(len({r[-1] for r in runs}), 9)
        _, runs = mp.expand_plan(PLAN, [0, 1], "p1", eps=[0.031373], select=["DKD"], epochs=5, lr=1e-4)
        self.assertEqual([r[0] for r in runs], ["/kd_final/B_DKD_0_0.0"] * 2)
        self.assertTrue(all("atepochs=5_lr=0.0001_" in r[-1] for r in runs))
        self.assertEqual({r[4] for r in runs}, {0, 1})
        with self.assertRaises(ValueError):
            mp.expand_plan(PLAN, [0], "v1", select=["nothing"])


class TestSourcesAliasMask(unittest.TestCase):
    def test_alias_and_mask_names(self):
        plan = dict(PLAN, Model_Path=["/kd_final/B_DKD_0_0.0",
                                      {"Path": "/pruning_final/very_long_sparsity=0.2_name", "Alias": "short_prune",
                                       "Keep_Zero_Mask": True}])
        self.assertEqual(mp.parse_sources(plan), [("/kd_final/B_DKD_0_0.0", None, False),
                                                  ("/pruning_final/very_long_sparsity=0.2_name", "short_prune", True)])
        _, runs = mp.expand_plan(plan, [0], "v1", eps=[0.031373])
        self.assertTrue(runs[0][-1].startswith("B_DKD_0_0.0_PGD_") and "_mask=" not in runs[0][-1])
        self.assertTrue(runs[1][-1].startswith("short_prune_PGD_") and "_advfrac=1_mask=kept_run=v1_" in runs[1][-1])
        self.assertEqual(runs[1][0], "/pruning_final/very_long_sparsity=0.2_name")     # real path kept
        _, runs = mp.expand_plan(plan, [0], "v1", select=["short_prune"])             # select matches alias
        self.assertEqual({r[0] for r in runs}, {"/pruning_final/very_long_sparsity=0.2_name"})
        for bad in ([{"Alias": "x"}], [{"Path": "/a", "Alias": "bad alias!"}], ["/a", "/a"]):
            with self.assertRaises(ValueError):
                mp.parse_sources(dict(PLAN, Model_Path=bad))

    def test_keep_zero_mask_survives_updates(self):
        torch.manual_seed(0)
        net = ResNet18(num_classes=10)
        with torch.no_grad():
            net.conv1.weight[:, 0] = 0.0                  # "pruned" entries
            net.fc.weight[0] = 0.0
        masks = mp.zero_masks(net)
        self.assertEqual(set(masks), {"conv1.weight", "fc.weight"})
        s0 = mp.conv_linear_sparsity(net)
        opt = mp.keep_zero_mask(torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4), net, masks)
        x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))
        for _ in range(3):
            opt.zero_grad()
            nn.CrossEntropyLoss()(net(x), y).backward()
            opt.step()
        self.assertTrue(torch.all(net.conv1.weight[:, 0] == 0) and torch.all(net.fc.weight[0] == 0))
        self.assertAlmostEqual(mp.conv_linear_sparsity(net), s0)
        self.assertGreater(net.conv1.weight[:, 1].abs().sum().item(), 0)


class TestStateAndSplit(unittest.TestCase):
    def test_load_state_unwraps_normalized_dict(self):
        net = ResNet18(num_classes=100)
        wrapped = NormalizedModel(net, (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        with tempfile.TemporaryDirectory() as d:
            for obj, tag in [(wrapped, "w"), (net, "b")]:
                torch.save(obj.state_dict(), Path(d) / f"{tag}.pth")
                state = mp.load_state(Path(d) / f"{tag}.pth")
                self.assertEqual(list(state.keys()), list(net.state_dict().keys()))

    def test_disjoint_split_check(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                idx = Path("Indices/X")
                idx.mkdir(parents=True)
                np.save(idx / "group_A_4_seed42.npy", np.array([0, 1, 2, 3]))
                np.save(idx / "group_B_25000_0.0_4_seed42.npy", np.array([4, 5, 6, 7]))
                self.assertEqual(mp.check_disjoint_split({"name": "X", "group_size": 4}), 4)
                np.save(idx / "group_B_25000_0.0_4_seed42.npy", np.array([3, 5, 6, 7]))
                with self.assertRaises(ValueError):
                    mp.check_disjoint_split({"name": "X", "group_size": 4})
            finally:
                os.chdir(cwd)


class TestJointPolicy(unittest.TestCase):
    def test_config_policies(self):
        self.assertEqual(mp.parse_posthoc_config(dict(PLAN, AdversarialTraining={"lambda": 1.0}))[3], "joint")
        self.assertEqual(mp.parse_posthoc_config(PLAN)[3], "frozen")
        for bad in ({"bn_policy": "clean_eval"}, {"bn_policy": "joint", "lambda": 0.0}):
            with self.assertRaises(ValueError):
                mp.parse_posthoc_config(dict(PLAN, AdversarialTraining=bad))

    def test_joint_loss_is_additive_objective(self):
        """With x_adv == x the 2B batch has the B batch's statistics: loss == (1 + lambda) * CE(x)."""
        torch.manual_seed(0)
        net = ResNet18(num_classes=100)
        torch.manual_seed(0)
        net2 = ResNet18(num_classes=100)                              # identical initialisation
        model = NormalizedModel(net, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 100, (8,))
        out = mp.joint_adversarial_step(model, x, y, nn.CrossEntropyLoss(), lambda m, a, b: a.clone(), 1.0, 0.5)
        m2 = NormalizedModel(net2, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)).train()
        ref = nn.CrossEntropyLoss()(m2(x), y)
        self.assertAlmostEqual(out["loss"].item(), 1.5 * ref.item(), places=5)
        self.assertTrue(torch.allclose(out["clean_logits"], out["adv_logits"], atol=1e-6))

    def test_adv_fraction_step(self):
        """k = B/4 adversarial copies: loss == CE(clean B) + lambda * CE(adv k) from one 5B/4 forward."""
        torch.manual_seed(0)
        net = ResNet18(num_classes=100)
        model = NormalizedModel(net, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 100, (8,))
        seen = {}
        def gen(m, xa, ya):
            seen["n"] = xa.size(0)
            return (xa + 0.01).clamp(0, 1)
        out = mp.joint_adversarial_step(model, x, y, nn.CrossEntropyLoss(), gen, 1.0, 0.5, adv_fraction=0.25)
        self.assertEqual(seen["n"], 2)
        self.assertEqual(tuple(out["adv_logits"].shape), (2, 100))
        self.assertTrue(torch.equal(out["adv_target"], y[:2]))
        ref = nn.CrossEntropyLoss()(out["clean_logits"], y) + 0.5 * nn.CrossEntropyLoss()(out["adv_logits"], y[:2])
        self.assertAlmostEqual(out["loss"].item(), ref.item(), places=6)
        self.assertEqual(mp.parse_adv_fraction(dict(PLAN, AdversarialTraining={"adv_fraction": 0.25})), 0.25)
        for bad in ({"adv_fraction": 0.0}, {"adv_fraction": 0.5, "bn_policy": "frozen"}):
            with self.assertRaises(ValueError):
                mp.parse_adv_fraction(dict(PLAN, AdversarialTraining=bad))
        _, runs = mp.expand_plan(dict(PLAN, AdversarialTraining={"adv_fraction": 0.25}), [0], "v1", select=["DKD"])
        self.assertTrue(all("_bn=joint_" in r[-1] and "_advfrac=0.25_" in r[-1] for r in runs))

    def test_joint_epoch_updates_running_stats(self):
        torch.manual_seed(0)
        net = ResNet18(num_classes=100)
        model = NormalizedModel(net, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        stats0 = mp.bn_running_stats(net)
        data = TensorDataset(torch.rand(16, 3, 32, 32), torch.randint(0, 100, (16,)))
        gen = make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 2})
        tr = mp.posthoc_one_epoch(model, net, DataLoader(data, batch_size=8), torch.optim.SGD(net.parameters(), lr=0.01),
                                  nn.CrossEntropyLoss(), gen, 1.0, 1.0, "cpu", None, "joint")
        self.assertFalse(all(torch.equal(stats0[k], v) for k, v in mp.bn_running_stats(net).items()))
        self.assertEqual(set(tr), {"train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
                                   "train_clean_acc", "train_adv_acc"})


class TestFrozenEpoch(unittest.TestCase):
    def test_frozen_epoch_trains_weights_not_running_stats(self):
        torch.manual_seed(0)
        net = ResNet18(num_classes=100)
        net.eval()
        with torch.no_grad():                      # non-trivial running statistics, as in a trained source
            net.train(); net(torch.rand(16, 3, 32, 32) * 2 - 1); net.eval()
        model = NormalizedModel(net, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        stats0 = mp.bn_running_stats(net)
        w0 = net.fc.weight.detach().clone()
        data = TensorDataset(torch.rand(16, 3, 32, 32), torch.randint(0, 100, (16,)))
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        gen = make_adv_generator(pgd_attack_v2, {"eps": 8 / 255, "steps": 2})
        tr = ft_at_one_epoch(model, net, DataLoader(data, batch_size=8), opt, nn.CrossEntropyLoss(), gen,
                             "FT-AL", 1.0, 1.0, "cpu", None, "frozen")
        stats1 = mp.bn_running_stats(net)
        self.assertTrue(all(torch.equal(stats0[k], stats1[k]) for k in stats0))
        self.assertGreater((net.fc.weight - w0).abs().max().item(), 0)
        self.assertTrue(np.isfinite(tr["train_loss"]) and np.isfinite(tr["train_adv_acc"]))


if __name__ == "__main__":
    unittest.main()
