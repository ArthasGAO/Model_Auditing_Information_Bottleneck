"""Small CPU integration tests for main_at_best.

All checkpoints, index caches and CSVs are temporary; no real model, no GPU,
no writes under saved_models / saved_logs / Indices.
"""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Subset

import calculate_MI_at as at
import MI_check as upstream
from util_adv import NormalizedModel


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)
BASE = "CIFAR-100_ResNet-18_400_Knockoff_Same100_Same18_0_1.0"
AT_TAIL = "PGD_eps=0.031373_steps=10_bn=adv_atn=400_atepochs=30_run=v1"
SCENARIO = "CIFAR-100_ResNet-18_400_Knockoff_Same100_Same18"


def at_name(seed=0, mix="off"):
    return f"{BASE}_{AT_TAIL}_atseed={seed}_mix={mix}"


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(12, 100)

    def forward(self, x):
        return self.fc(x.flatten(1))


class Inputs:
    def __init__(self):
        self.targets = np.repeat(np.arange(100), 4)
        self.x = torch.randn(400, 3, 2, 2, generator=torch.Generator().manual_seed(17)) * 3

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.x[index], int(self.targets[index])


class Data:
    """Mimics the CIFAR wrapper: normalized and raw views share indices."""

    def __init__(self):
        self.in_sample_set = Inputs()
        self.raw_train_clean_set = self.in_sample_set
        self.raw_test_set = Subset(self.in_sample_set, list(range(0, 400, 4)))
        self.test_set = self.raw_test_set
        self.mean, self.std = MEAN, STD

    def subset(self, split, indices, clean=False):
        assert split == "raw_train_clean", split
        return Subset(self.raw_train_clean_set, indices)


class AtBestMITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / "plan.yaml"
        self.plan.write_text("at plan identity\n", encoding="utf-8")
        self.csv = self.root / "at_best.csv"
        self.config = {
            "Model": "ResNet-18",
            "Dataset": {"name": "CIFAR-100", "group_size": 400},
        }
        self.data = Data()
        self.checkpoints = {k: self.make_checkpoint(at_name(), k)
                            for k in ("best_clean", "best_rob")}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for key, value in {"device": "cpu", "AT_BEST_NUM_WORKERS": 0,
                           "AT_BEST_BATCH_SIZE": 64}.items():
            self.stack.enter_context(patch.object(at, key, value))
        self.stack.enter_context(patch.object(at, "set_seed"))
        self.stack.enter_context(patch.object(at, "process_yaml_file", return_value=self.config))
        self.stack.enter_context(patch.object(
            at, "build_dataset_from_yaml", return_value=(self.data, 100, 400)))
        self.builder = self.stack.enter_context(patch.object(
            at, "build_model", side_effect=lambda name, n: TinyNet()))
        self.inference = self.stack.enter_context(patch.object(
            at, "collect_logits", wraps=at.collect_logits))
        self.measure = self.stack.enter_context(patch.object(
            at, "mi_from_logits", wraps=at.mi_from_logits))

    def make_checkpoint(self, folder, kind, *, bare=False, mean=MEAN, std=STD):
        path = self.root / "models" / folder / f"{kind}_epoch.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        net = TinyNet()
        with torch.no_grad():
            net.fc.weight.mul_(4 if kind == "best_clean" else 6)
        state = net.state_dict() if bare else NormalizedModel(net, mean, std).state_dict()
        torch.save(state, path)
        return path

    def run_best(self, folder=None, **overrides):
        args = dict(
            model_dir=self.root / "models", master_csv_path=self.csv,
            index_dir=self.root / "indices", verbose_dir=self.root / "verbose",
            in_size_rates=[0.625, 1.0], num_intervals_list=[5, 10],
        )
        if "in_sizes" in overrides and "in_size_rates" not in overrides:
            args.pop("in_size_rates")
        args.update(overrides)
        return at.main_at_best(folder or at_name(), self.plan, **args)

    def rows(self):
        with self.csv.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    # ------------------------------------------------------------------
    def test_scenario_name_parsing(self):
        parsed = at.parse_at_scenario_name(at_name(seed=2, mix="0.8"))
        self.assertEqual(parsed["Scenario"], SCENARIO)
        self.assertEqual(parsed["base_model"], BASE)
        self.assertEqual((parsed["attack"], parsed["eps"], parsed["steps"]),
                         ("PGD", "0.031373", "10"))
        self.assertEqual((parsed["bn"], parsed["at_size"], parsed["at_epochs"]),
                         ("adv", 400, 30))
        self.assertEqual((parsed["run_tag"], parsed["at_seed"], parsed["mix"]),
                         ("v1", 2, "0.8"))
        self.assertEqual(parsed["rate"], 1.0)
        # bn=clean_eval keeps its underscore; the atn= marker ends the field.
        mixed = at.parse_at_scenario_name(
            f"{BASE}_PGD_eps=0.01_steps=7_bn=clean_eval_atn=400_atepochs=50_atseed=1_mix=0.5")
        self.assertEqual((mixed["bn"], mixed["run_tag"], mixed["at_epochs"]),
                         ("clean_eval", "", 50))
        for bad in ["no_markers_at_all", f"{BASE}_PGD_eps=0.01_bn=adv_atn=400_atseed=0",
                    "PGD_eps=0.01_bn=adv_atn=400_atepochs=30_atseed=0_mix=off"]:
            with self.assertRaises(ValueError):
                at.parse_at_scenario_name(bad)

    def test_discovery_filters_and_reports_incomplete_folders(self):
        self.make_checkpoint(at_name(seed=1), "best_clean")   # best_rob missing
        found = at.discover_at_models(self.root / "models")
        self.assertEqual([n for n, _ in found], [at_name()])
        found = at.discover_at_models(self.root / "models", ckpt_kinds=["best_clean"])
        self.assertEqual([n for n, _ in found], [at_name(), at_name(seed=1)])
        found = at.discover_at_models(self.root / "models", selectors=["atseed=1"],
                                      ckpt_kinds=["best_clean"])
        self.assertEqual([n for n, _ in found], [at_name(seed=1)])
        with self.assertRaises(FileNotFoundError):
            at.discover_at_models(self.root / "absent")

    def test_both_checkpoints_identity_columns_and_mi_values(self):
        self.assertEqual(self.run_best(record_verbose=True), 8)   # 2 kinds x 2 sizes x 2 bins
        rows = self.rows()
        self.assertEqual({r["epoch"] for r in rows}, {"best_clean", "best_rob"})
        self.assertEqual({r["model_name"] for r in rows},
                         {f"{at_name()}_ckpt=best_clean", f"{at_name()}_ckpt=best_rob"})
        for row in rows:
            self.assertEqual(row["Scenario"], SCENARIO)
            self.assertEqual(row["at_scenario"], at_name())
            self.assertEqual(row["base_model"], BASE)
            self.assertEqual((row["seed"], row["at_seed"], row["rate"]), ("0", "0", "1.0"))
            self.assertEqual((row["attack"], row["eps"], row["steps"]), ("PGD", "0.031373", "10"))
            self.assertEqual((row["bn"], row["mix"], row["run_tag"]), ("adv", "off", "v1"))
            self.assertEqual((row["at_size"], row["at_epochs"]), ("400", "30"))
            self.assertEqual((row["training_size"], row["family"]), ("400", "cnn"))
            self.assertEqual((row["subset_seed"], row["group_seed"]), ("42", "42"))
            self.assertEqual(row["out_size"], "100")
            self.assertEqual(row["ckpt_kind"], row["epoch"])
        self.assertEqual({r["in_size_rate"] for r in rows}, {"0.625", "1"})
        # 2 checkpoints x (one Out + two In)
        self.assertEqual(self.inference.call_count, 6)
        self.assertEqual(len(list((self.root / "verbose").rglob("*.npz"))), 12)

        # The In MI must match a direct upstream computation through the wrapper.
        with np.load(self.root / "indices/CIFAR-100/nested_subsets_seed42.npz") as cache:
            indices = cache["size_250"]
        net = NormalizedModel(TinyNet(), MEAN, STD)
        net.load_state_dict(torch.load(self.checkpoints["best_clean"], weights_only=True))
        net.eval()
        with torch.no_grad():
            logits = net(self.data.in_sample_set.x[indices])
        labels = torch.nn.functional.one_hot(
            torch.tensor(self.data.in_sample_set.targets[indices]), 100).float()
        expected = upstream.mi_from_logits(logits, labels, 5)
        self.assertGreater(expected[0], 0)
        row = next(r for r in rows if r["in_size"] == "250" and r["bins"] == "5"
                   and r["epoch"] == "best_clean")
        np.testing.assert_allclose(
            [float(row["I(X;T)-In"]), float(row["I(T;Y)-In"])], expected, atol=1e-6)
        # The two checkpoints are different models, so their MI differs.
        other = next(r for r in rows if r["in_size"] == "250" and r["bins"] == "5"
                     and r["epoch"] == "best_rob")
        self.assertNotAlmostEqual(float(row["I(X;T)-In"]), float(other["I(X;T)-In"]))

    def test_uses_shared_subset_builder_with_remainder_allocation(self):
        self.run_best(ckpt_kinds=["best_clean"])
        with np.load(self.root / "indices/CIFAR-100/nested_subsets_seed42.npz") as cache:
            self.assertEqual(sorted(int(k[5:]) for k in cache.files), [250, 400])
            counts = np.bincount(
                self.data.in_sample_set.targets[cache["size_250"]], minlength=100)
        np.testing.assert_array_equal(np.sort(counts), [2] * 50 + [3] * 50)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            at.create_nested_balanced_subsets(
                self.data.in_sample_set, np.arange(400), self.root / "legacy",
                [250], num_classes=100, seed=42)

    def test_resume_skips_complete_and_extends_reusing_out_mi(self):
        self.assertEqual(self.run_best(), 8)
        before = self.csv.read_bytes()
        self.inference.reset_mock()
        self.builder.reset_mock()
        self.assertEqual(self.run_best(), 0)
        self.assertEqual(self.csv.read_bytes(), before)
        self.inference.assert_not_called()
        self.builder.assert_not_called()
        # One checkpoint, one new rate: Out MI is reused, so one In inference.
        self.assertEqual(self.run_best(in_size_rates=[0.5], ckpt_kinds=["best_rob"]), 2)
        self.assertEqual(self.inference.call_count, 1)
        self.assertTrue(self.csv.read_bytes().startswith(before))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_best(skip_existing=False)

    def test_rejects_duplicate_identity_drift_nonfinite_and_stale_checkpoint(self):
        self.run_best(ckpt_kinds=["best_clean"])
        original = self.csv.read_bytes()
        with self.csv.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=at.AT_BEST_CSV_COLUMNS).writerow(self.rows()[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_best(ckpt_kinds=["best_clean"])
        self.csv.write_bytes(original)
        for field, value, message in [("epoch", "99", "identity mismatch"),
                                      ("eps", "0.007843", "identity mismatch"),
                                      ("bn", "clean_eval", "identity mismatch"),
                                      ("at_seed", "1", "identity mismatch"),
                                      ("I(T;Y)-In", "nan", "Nonfinite"),
                                      ("in_size_rate", "0.01", "fraction"),
                                      ("I(X;T)-Out", "999", "Inconsistent")]:
            rows = self.rows()
            rows[0][field] = value
            with self.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=at.AT_BEST_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, message):
                self.run_best(ckpt_kinds=["best_clean"])
            self.csv.write_bytes(original)
        self.checkpoints["best_clean"].write_bytes(
            self.checkpoints["best_clean"].read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_best(ckpt_kinds=["best_clean"])
        self.assertEqual(self.csv.read_bytes(), original)

    def test_accepts_bare_backbone_and_rejects_normalization_mismatch(self):
        folder = at_name(seed=1)
        self.make_checkpoint(folder, "best_clean", bare=True)
        self.assertEqual(self.run_best(folder, ckpt_kinds=["best_clean"],
                                       in_size_rates=[1.0], num_intervals_list=[5]), 1)
        self.assertEqual(self.rows()[0]["at_seed"], "1")
        folder = at_name(seed=2)
        self.make_checkpoint(folder, "best_clean", mean=(0.1, 0.2, 0.3))
        with self.assertRaisesRegex(ValueError, "differs from dataset mean"):
            self.run_best(folder, ckpt_kinds=["best_clean"],
                          in_size_rates=[1.0], num_intervals_list=[5])

    def test_rejects_legacy_thin_csv_header_and_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            self.run_best(at_name(seed=3))
        self.assertFalse(self.csv.exists())
        self.assertFalse((self.root / "indices").exists())
        self.builder.assert_not_called()
        self.csv.write_text(",".join(at.MASTER_CSV_COLUMNS) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "header"):
            self.run_best()

    # ---- plan-driven selection ---------------------------------------
    def plan_config(self, paths=None, eps=None, seeds_note=None):
        return {
            "Model_Path": paths or ["/victim_models/CIFAR-100_ResNet-18_400_Knockoff_Same100_Same18_0_1.0"],
            "Dataset": {"name": "CIFAR-100", "group_size": 400},
            "Model": "ResNet-18",
            "AdversarialTraining": {"use_mixed": False, "mix_rate": 0.0},
            "Attack": {"name": "PGD", "eps": eps if eps is not None else [0.031373], "steps": 10},
            "Optimizer": {"name": "SGD", "params": {}, "Epochs": 30},
        }

    def test_expand_at_plan_rebuilds_training_names(self):
        self.config.clear()
        self.config.update(self.plan_config(eps=[0.007843, 0.031373]))
        names = at.expand_at_plan(self.plan, at_seeds=[0, 1], run_tag="v1")
        self.assertEqual(len(names), 4)          # 1 path x 2 eps x 2 seeds
        self.assertIn(at_name(seed=0), names)
        self.assertIn(at_name(seed=1), names)
        for n in names:
            parsed = at.parse_at_scenario_name(n)   # every name must round-trip
            self.assertEqual(parsed["base_model"], BASE)
            self.assertEqual((parsed["at_size"], parsed["at_epochs"]), (400, 30))
            self.assertEqual((parsed["run_tag"], parsed["mix"]), ("v1", "off"))
        self.assertEqual(len(at.expand_at_plan(self.plan, at_seeds=[0])), 2)
        for bad in ([], [0, 0], [-1], ["a"]):
            with self.assertRaisesRegex(ValueError, "at_seeds"):
                at.expand_at_plan(self.plan, at_seeds=bad)
        self.config["Model_Path"] = []
        with self.assertRaisesRegex(ValueError, "no Model_Path"):
            at.expand_at_plan(self.plan, at_seeds=[0])

    def test_plan_selection_skips_untrained_and_honours_selectors(self):
        self.config.clear()
        self.config.update(self.plan_config(eps=[0.007843, 0.031373]))
        # Only the eps=0.031373 seeds 0 and 1 exist on disk (setUp made seed 0).
        self.make_checkpoint(at_name(seed=1), "best_clean")
        self.make_checkpoint(at_name(seed=1), "best_rob")
        found = at.models_from_at_plans(
            plans=[self.plan], at_seeds=[0, 1], run_tag="v1",
            model_dir=self.root / "models")
        self.assertEqual({n for n, _, _ in found}, {at_name(0), at_name(1)})
        self.assertEqual({str(p) for _, _, p in found}, {str(self.plan)})
        with self.assertRaisesRegex(FileNotFoundError, "no checkpoint"):
            at.models_from_at_plans(
                plans=[self.plan], at_seeds=[0, 1], run_tag="v1",
                model_dir=self.root / "models", missing="raise")
        narrowed = at.models_from_at_plans(
            plans=[self.plan], at_seeds=[0, 1], run_tag="v1",
            model_dir=self.root / "models", selectors=["atseed=1"])
        self.assertEqual([n for n, _, _ in narrowed], [at_name(1)])
        with self.assertRaisesRegex(ValueError, "Two plans declare"):
            at.models_from_at_plans(
                plans=[self.plan, self.plan], at_seeds=[0], run_tag="v1",
                model_dir=self.root / "models")
        with self.assertRaisesRegex(ValueError, "missing must be"):
            at.models_from_at_plans(plans=[self.plan], missing="maybe")

    def test_sweep_is_plan_driven_not_directory_driven(self):
        self.config.clear()
        self.config.update(self.plan_config())
        # A model that exists on disk but is NOT declared by the plan.
        stray = at_name(seed=4)
        self.make_checkpoint(stray, "best_clean")
        self.make_checkpoint(stray, "best_rob")
        self.assertEqual(len(at.discover_at_models(self.root / "models")), 2)
        tally = at.run_at_best_sweep(
            plans=[self.plan], at_seeds=[0], run_tag="v1",
            model_dir=self.root / "models", master_csv_path=self.csv,
            index_dir=self.root / "indices", verbose_dir=self.root / "verbose",
            in_size_rates=[1.0], num_intervals_list=[5],
        )
        self.assertEqual(list(tally), [at_name(0)])          # stray not computed
        self.assertEqual({r["at_seed"] for r in self.rows()}, {"0"})
        # from_disk=True restores the old directory behaviour.
        tally = at.run_at_best_sweep(
            from_disk=True, yaml_file_path=self.plan,
            model_dir=self.root / "models", master_csv_path=self.csv,
            index_dir=self.root / "indices", verbose_dir=self.root / "verbose",
            in_size_rates=[1.0], num_intervals_list=[5],
        )
        self.assertEqual(sorted(tally), sorted([at_name(0), stray]))
        self.assertEqual({r["at_seed"] for r in self.rows()}, {"0", "4"})

    def test_grid_validation(self):
        with self.assertRaisesRegex(ValueError, "Choose in_sizes or in_size_rates"):
            self.run_best(in_sizes=[400], in_size_rates=[1.0])
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.run_best(in_sizes=[800])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.run_best(in_sizes=[0])
        with self.assertRaisesRegex(ValueError, "no silent rounding|Nonintegral"):
            self.run_best(in_size_rates=[0.001])
        with self.assertRaisesRegex(ValueError, "unique kinds"):
            self.run_best(ckpt_kinds=["best_clean", "best_clean"])
        self.assertFalse(self.csv.exists())


if __name__ == "__main__":
    unittest.main()
