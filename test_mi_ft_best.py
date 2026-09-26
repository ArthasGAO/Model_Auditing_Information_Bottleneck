"""Small CPU integration tests; all checkpoints, indices and CSVs are temporary."""
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

import calculate_MI_ft as ft
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


class BestMITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / "plan.yaml"
        self.plan.write_text("test plan identity\n", encoding="utf-8")
        self.csv = self.root / "best.csv"
        self.config = {
            "Scenario_Name": "CIFAR-100_Test", "Model": "ResNet-18",
            "Dataset": {"name": "CIFAR-100", "group_size": 400},
            "FT_Dataset": {"name": "unused", "group_size": 20},
        }
        self.data = Data()
        self.checkpoint = self.make_checkpoint("FT-AL")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for key, value in {"device": "cpu", "BEST_NUM_WORKERS": 0, "BEST_BATCH_SIZE": 64}.items():
            self.stack.enter_context(patch.object(ft, key, value))
        self.stack.enter_context(patch.object(ft, "set_seed"))
        self.stack.enter_context(patch.object(ft, "process_yaml_file", return_value=self.config))

        def setup(config):
            self.assertNotIn("FT_Dataset", config)
            return {"Dataset": self.data, "NumClasses": 100, "Model_Factory": TinyNet}

        self.setup_cnn = self.stack.enter_context(patch.object(ft, "process_experiment_ft_setup", side_effect=setup))
        self.setup_deit = self.stack.enter_context(patch.object(ft, "process_experiment_ft_setup_deit", side_effect=setup))
        self.inference = self.stack.enter_context(patch.object(ft, "collect_logits", wraps=upstream.collect_logits))
        self.measure = self.stack.enter_context(patch.object(ft, "mi_from_logits", wraps=upstream.mi_from_logits))

    def make_checkpoint(self, strategy, wrapper=None):
        name = f"CIFAR-100_Test_42_1.0_{strategy}_ftsize=20_ftseed=7"
        path = self.root / "models" / name / "best_epoch.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        net = TinyNet()
        with torch.no_grad():
            net.fc.weight.mul_(5)
        state = net.state_dict()
        torch.save({wrapper: state} if wrapper else state, path)
        return path

    def run_best(self, **overrides):
        args = dict(
            model_dir=self.root / "models", master_csv_path=self.csv,
            index_dir=self.root / "indices", verbose_dir=self.root / "verbose",
            strategies=["FT-AL"], in_size_rates=[0.625, 1.0], num_intervals_list=[5, 10],
        )
        args.update(overrides)
        return ft.main_best(42, 7, 1.0, self.plan, **args)

    def rows(self):
        with self.csv.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_real_group_cache_accepts_path_and_string_without_rewriting(self):
        cache_dir = self.root / "group_cache"
        group = ft.create_or_load_group_A(
            self.data.in_sample_set, cache_dir, group_size=400, num_classes=100,
        )
        cache = cache_dir / "group_A_400_seed42.npy"
        original = cache.read_bytes()
        self.assertEqual(len(set(group)), 400)
        for save_dir in (str(cache_dir), cache_dir):
            # A cached load must not inspect the dataset or resample indices.
            replay = ft.create_or_load_group_A(None, save_dir, group_size=400, num_classes=100)
            self.assertEqual(replay, group)
            self.assertEqual(cache.read_bytes(), original)

    def test_rate_grid_remainder_numerics_and_incremental_extension(self):
        self.assertEqual(self.run_best(record_verbose=True), 4)
        rows = self.rows()
        self.assertEqual({(r["in_size"], r["bins"]) for r in rows},
                         {("250", "5"), ("250", "10"), ("400", "5"), ("400", "10")})
        for row in rows:
            self.assertEqual((row["seed"], row["model_seed"], row["ft_seed"], row["epoch"]),
                             ("7", "42", "7", "best"))
            self.assertEqual(row["training_size"], "400")
            self.assertEqual(row["ft_size"], "20")
        self.assertEqual(self.inference.call_count, 3)  # one Out, two In
        self.assertEqual(len(list((self.root / "verbose").rglob("*.npz"))), 6)
        with np.load(self.root / "indices/CIFAR-100/nested_subsets_seed42.npz") as cache:
            indices = cache["size_250"]
            counts = np.bincount(self.data.in_sample_set.targets[indices], minlength=100)
            np.testing.assert_array_equal(np.sort(counts), [2] * 50 + [3] * 50)
        net = TinyNet()
        net.load_state_dict(torch.load(self.checkpoint, weights_only=True))
        logits = net(self.data.in_sample_set.x[indices]).detach()
        labels = torch.nn.functional.one_hot(torch.tensor(self.data.in_sample_set.targets[indices]), 100).float()
        expected = upstream.mi_from_logits(logits, labels, 5)
        self.assertGreater(expected[0], 0)
        np.testing.assert_allclose([float(rows[0]["I(X;T)-In"]), float(rows[0]["I(T;Y)-In"])], expected, atol=1e-6)
        before = self.csv.read_bytes()
        self.inference.reset_mock()
        self.setup_cnn.reset_mock()
        self.assertEqual(self.run_best(), 0)
        self.assertEqual(self.csv.read_bytes(), before)
        self.inference.assert_not_called()
        self.setup_cnn.assert_not_called()
        self.assertEqual(self.run_best(in_size_rates=[0.75]), 2)
        self.assertEqual(self.inference.call_count, 1)  # existing Out reused
        self.assertTrue(self.csv.read_bytes().startswith(before))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_best(skip_existing=False)

    def test_interrupted_write_resumes_only_missing_pairs_and_out_bins(self):
        append = ft._append_best_row
        calls = 0

        def interrupt(path, row):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            append(path, row)

        with patch.object(ft, "_append_best_row", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.run_best()
        first = self.csv.read_bytes()
        self.assertEqual(len(self.rows()), 1)
        self.measure.reset_mock()
        self.assertEqual(self.run_best(), 3)
        self.assertTrue(self.csv.read_bytes().startswith(first))
        out_bins = [c.kwargs["num_intervals"] for c in self.measure.call_args_list if len(c.args[0]) == 100]
        self.assertEqual(out_bins, [10])
        self.assertEqual(len(self.rows()), 4)

    def test_rejects_stale_checkpoint_duplicate_invalid_and_mixed_epoch_rows(self):
        self.run_best()
        original = self.csv.read_bytes()
        with self.csv.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=ft.BEST_CSV_COLUMNS).writerow(self.rows()[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_best()
        self.csv.write_bytes(original)
        for field, value, message in [("epoch", "99", "identity mismatch"),
                                       ("I(T;Y)-In", "nan", "Nonfinite"),
                                       ("in_size_rate", "0.01", "fraction"),
                                       ("I(X;T)-Out", "999", "Inconsistent")]:
            rows = self.rows()
            rows[0][field] = value
            with self.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=ft.BEST_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, message):
                self.run_best()
            self.csv.write_bytes(original)
        self.checkpoint.write_bytes(self.checkpoint.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            self.run_best()
        self.assertEqual(self.csv.read_bytes(), original)

    def test_preflights_all_strategies_and_rejects_legacy_csv(self):
        with self.assertRaises(FileNotFoundError):
            self.run_best(strategies=["FT-AL", "FT-LL"])
        self.assertFalse(self.csv.exists())
        self.assertFalse((self.root / "indices").exists())
        self.setup_cnn.assert_not_called()
        self.csv.write_text("Scenario,seed,epoch\nlegacy,7,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "header"):
            self.run_best()

    def test_deit_dispatch_and_both_checkpoint_wrappers(self):
        self.config["Model"] = {"model_name": "deit_tiny_patch16_224", "pretrained": True}
        for strategy, wrapper in [("FT-LL", "model"), ("RT-AL", "state_dict")]:
            self.make_checkpoint(strategy, wrapper)
        self.assertEqual(self.run_best(strategies=["FT-LL", "RT-AL"], in_size_rates=[0.625]), 4)
        self.setup_cnn.assert_not_called()
        self.setup_deit.assert_called_once()
        self.assertFalse(self.setup_deit.call_args.args[0]["Model"]["pretrained"])
        self.assertTrue(self.config["Model"]["pretrained"])
        self.assertEqual({r["family"] for r in self.rows()}, {"deit"})


if __name__ == "__main__":
    unittest.main()
