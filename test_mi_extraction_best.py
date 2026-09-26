"""Small CPU integration tests for main_extraction_best.

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

import calculate_MI_extraction as ext
import MI_check as upstream


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 100)

    def forward(self, x):
        return self.fc(x)


class Inputs:
    def __init__(self):
        self.targets = np.repeat(np.arange(100), 4)
        self.x = torch.randn(400, 4, generator=torch.Generator().manual_seed(17)) * 4

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.x[index], int(self.targets[index])


class Data:
    def __init__(self):
        self.in_sample_set = Inputs()
        self.test_set = Subset(self.in_sample_set, list(range(0, 400, 4)))

    def subset(self, split, indices, clean):
        assert split == "train" and clean is True
        return Subset(self.in_sample_set, indices)


SCENARIO = "CIFAR-100_ResNet-18_400_Knockoff_Same100_Same18"


class ExtractionBestMITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / "plan.yaml"
        self.plan.write_text("extraction plan identity\n", encoding="utf-8")
        self.csv = self.root / "extraction_best.csv"
        self.config = {
            "Scenario_Name": SCENARIO,
            "Victim": {"Model_Name": "CIFAR-100_ResNet-18_400", "Model": "ResNet-18",
                       "Dataset": {"name": "CIFAR-100", "group_size": 400}},
            "Substitute": {"Model": "ResNet-18"},
            "Auxiliary_Dataset": {"name": "CIFAR-100", "group_size": 400},
            "Knockoff": {"sampling_size": 1.0},
        }
        self.data = Data()
        self.checkpoint = self.make_checkpoint(0)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for key, value in {"device": "cpu", "EXTRACTION_BEST_NUM_WORKERS": 0,
                           "EXTRACTION_BEST_BATCH_SIZE": 64}.items():
            self.stack.enter_context(patch.object(ext, key, value))
        self.stack.enter_context(patch.object(ext, "set_seed"))
        self.stack.enter_context(patch.object(ext, "process_yaml_file", return_value=self.config))
        self.stack.enter_context(patch.object(
            ext, "build_dataset_from_yaml", return_value=(self.data, 100, 400)))
        self.builder = self.stack.enter_context(patch.object(
            ext, "build_model", side_effect=lambda name, n: TinyNet()))
        self.inference = self.stack.enter_context(patch.object(
            ext, "collect_logits", wraps=upstream.collect_logits))
        self.measure = self.stack.enter_context(patch.object(
            ext, "mi_from_logits", wraps=upstream.mi_from_logits))

    def make_checkpoint(self, seed, wrapper=None):
        path = self.root / "models" / f"{SCENARIO}_{seed}_1.0" / "best_epoch.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        net = TinyNet()
        with torch.no_grad():
            net.fc.weight.mul_(5)
        state = net.state_dict()
        torch.save({wrapper: state} if wrapper else state, path)
        return path

    def run_best(self, seed=0, **overrides):
        args = dict(
            model_dir=self.root / "models", master_csv_path=self.csv,
            index_dir=self.root / "indices", verbose_dir=self.root / "verbose",
            in_size_rates=[0.625, 1.0], num_intervals_list=[5, 10],
        )
        # The default grid is rate-based; an explicit in_sizes replaces it, since
        # main_extraction_best rejects receiving both.
        if "in_sizes" in overrides and "in_size_rates" not in overrides:
            args.pop("in_size_rates")
        args.update(overrides)
        return ext.main_extraction_best(seed, self.plan, **args)

    def rows(self):
        with self.csv.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    # ------------------------------------------------------------------
    def test_grid_identity_columns_and_mi_values(self):
        self.assertEqual(self.run_best(record_verbose=True), 4)
        rows = self.rows()
        self.assertEqual({(r["in_size"], r["bins"]) for r in rows},
                         {("250", "5"), ("250", "10"), ("400", "5"), ("400", "10")})
        for row in rows:
            self.assertEqual(row["Scenario"], SCENARIO)
            self.assertEqual((row["seed"], row["rate"], row["epoch"]), ("0", "1.0", "best"))
            self.assertEqual(row["model_name"], f"{SCENARIO}_0_1.0")
            self.assertEqual(row["attack"], "Knockoff")
            self.assertEqual(row["victim_model"], "ResNet-18")
            self.assertEqual(row["substitute_model"], "ResNet-18")
            self.assertEqual(row["aux_dataset"], "CIFAR-100")
            self.assertEqual((row["training_size"], row["family"]), ("400", "cnn"))
            self.assertEqual((row["subset_seed"], row["group_seed"]), ("42", "42"))
            self.assertEqual(row["out_size"], "100")
            self.assertEqual(row["checkpoint_sha256"],
                             ext._extraction_file_sha256(self.checkpoint))
        self.assertEqual({r["in_size_rate"] for r in rows}, {"0.625", "1"})
        self.assertEqual(self.inference.call_count, 3)   # one Out, two In
        self.assertEqual(len(list((self.root / "verbose").rglob("*.npz"))), 6)

        # The In MI must match a direct upstream computation on the same probe.
        with np.load(self.root / "indices/CIFAR-100/nested_subsets_seed42.npz") as cache:
            indices = cache["size_250"]
        net = TinyNet()
        net.load_state_dict(torch.load(self.checkpoint, weights_only=True))
        logits = net(self.data.in_sample_set.x[indices]).detach()
        labels = torch.nn.functional.one_hot(
            torch.tensor(self.data.in_sample_set.targets[indices]), 100).float()
        expected = upstream.mi_from_logits(logits, labels, 5)
        self.assertGreater(expected[0], 0)
        row = next(r for r in rows if r["in_size"] == "250" and r["bins"] == "5")
        np.testing.assert_allclose(
            [float(row["I(X;T)-In"]), float(row["I(T;Y)-In"])], expected, atol=1e-6)

    def test_uses_shared_subset_builder_with_remainder_allocation(self):
        """250 is not divisible by 100; the local legacy builder rejects it."""
        self.run_best()
        with np.load(self.root / "indices/CIFAR-100/nested_subsets_seed42.npz") as cache:
            self.assertEqual(sorted(int(k[5:]) for k in cache.files), [250, 400])
            counts = np.bincount(
                self.data.in_sample_set.targets[cache["size_250"]], minlength=100)
        np.testing.assert_array_equal(np.sort(counts), [2] * 50 + [3] * 50)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            ext.create_nested_balanced_subsets(
                self.data.in_sample_set, np.arange(400), self.root / "legacy",
                [250], num_classes=100, seed=42)

    def test_resume_skips_complete_and_extends_reusing_out_mi(self):
        self.assertEqual(self.run_best(), 4)
        before = self.csv.read_bytes()
        self.inference.reset_mock()
        self.builder.reset_mock()
        self.assertEqual(self.run_best(), 0)
        self.assertEqual(self.csv.read_bytes(), before)
        self.inference.assert_not_called()
        self.builder.assert_not_called()           # no model built when nothing is missing
        # A new rate reuses the Out MI already in the CSV: one In inference only.
        self.assertEqual(self.run_best(in_size_rates=[0.5]), 2)
        self.assertEqual(self.inference.call_count, 1)
        self.assertTrue(self.csv.read_bytes().startswith(before))
        self.assertEqual({r["in_size"] for r in self.rows()}, {"200", "250", "400"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_best(skip_existing=False)

    def test_rejects_duplicate_identity_drift_nonfinite_and_stale_checkpoint(self):
        self.run_best()
        original = self.csv.read_bytes()
        with self.csv.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=ext.EXTRACTION_BEST_CSV_COLUMNS).writerow(
                self.rows()[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_best()
        self.csv.write_bytes(original)
        for field, value, message in [("epoch", "99", "identity mismatch"),
                                      ("substitute_model", "VGG16", "identity mismatch"),
                                      ("attack", "JBA", "identity mismatch"),
                                      ("I(T;Y)-In", "nan", "Nonfinite"),
                                      ("in_size_rate", "0.01", "fraction"),
                                      ("I(X;T)-Out", "999", "Inconsistent")]:
            rows = self.rows()
            rows[0][field] = value
            with self.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=ext.EXTRACTION_BEST_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, message):
                self.run_best()
            self.csv.write_bytes(original)
        self.checkpoint.write_bytes(self.checkpoint.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_best()
        self.assertEqual(self.csv.read_bytes(), original)

    def test_rejects_legacy_thin_csv_header(self):
        self.csv.write_text(
            ",".join(ext.MASTER_CSV_COLUMNS) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "header"):
            self.run_best()

    def test_missing_checkpoint_touches_nothing(self):
        with self.assertRaises(FileNotFoundError):
            self.run_best(seed=3)
        self.assertFalse(self.csv.exists())
        self.assertFalse((self.root / "indices").exists())
        self.builder.assert_not_called()

    def test_rejects_deit_substitute_and_ambiguous_attack_block(self):
        self.config["Substitute"]["Model"] = "DeiT"
        with self.assertRaisesRegex(ValueError, "calculate_MI_extraction_deit"):
            self.run_best()
        self.config["Substitute"]["Model"] = "ResNet-18"
        self.config["JBA"] = {}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_best()
        del self.config["JBA"]
        del self.config["Knockoff"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_best()
        self.assertFalse(self.csv.exists())

    def test_grid_validation_and_checkpoint_wrappers(self):
        with self.assertRaisesRegex(ValueError, "Choose in_sizes or in_size_rates"):
            self.run_best(in_sizes=[400], in_size_rates=[1.0])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.run_best(in_sizes=[0])
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.run_best(in_sizes=[800])
        with self.assertRaisesRegex(ValueError, "no silent rounding|Nonintegral"):
            self.run_best(in_size_rates=[0.001])
        for seed, wrapper in [(1, "model"), (2, "state_dict")]:
            self.make_checkpoint(seed, wrapper)
            self.assertEqual(self.run_best(seed=seed, in_size_rates=[1.0],
                                           num_intervals_list=[5]), 1)
        self.assertEqual({r["seed"] for r in self.rows()}, {"1", "2"})

    def test_explicit_in_sizes_and_aux_from_proxy_block(self):
        self.config["Proxy"] = {"Dataset": {"name": "CIFAR-10"}}
        del self.config["Auxiliary_Dataset"]
        self.assertEqual(self.run_best(in_sizes=[400], num_intervals_list=[5]), 1)
        row = self.rows()[0]
        self.assertEqual(row["aux_dataset"], "CIFAR-10")
        self.assertEqual((row["in_size"], row["in_size_rate"]), ("400", "1"))


if __name__ == "__main__":
    unittest.main()
