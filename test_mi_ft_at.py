"""CPU tests for calculate_MI_ft_at.py naming, columns and CSV identity rules (temp dirs only).

Run:  python test_mi_ft_at.py
"""
import csv
import tempfile
import unittest
from pathlib import Path

import calculate_MI_ft_at as mft
from calculate_MI_ft import BEST_CSV_COLUMNS


PLAN = {
    "Scenario_Name": "CIFAR-10_ResNet-18_25000_Same_25000",
    "AdversarialTraining": {"form": "additive", "lambda": 1.0, "bn_policy": "frozen"},
    "Attack": {"name": "PGD", "eps": [0.007843, 0.031373], "steps": 10},
}


class MIFtAtTests(unittest.TestCase):

    def test_columns_extend_ft_table(self):
        self.assertEqual(mft.FT_AT_CSV_COLUMNS[:len(BEST_CSV_COLUMNS)], BEST_CSV_COLUMNS)
        self.assertEqual(mft.FT_AT_CSV_COLUMNS[len(BEST_CSV_COLUMNS):],
                         ["base_model_name", "attack", "eps", "steps", "lambda", "bn_policy", "run_tag"])

    def test_names_match_training_names(self):
        base = mft.ft_base_name(PLAN["Scenario_Name"], 42, 1.0, "FT-AL", 25000, 0)
        self.assertEqual(base, "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_FT-AL_ftsize=25000_ftseed=0")
        suffixes = mft.plan_suffixes(PLAN, "v1")
        self.assertEqual(len(suffixes), 2)
        self.assertEqual(base + suffixes[1][0],
                         "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_FT-AL_ftsize=25000_ftseed=0"
                         "_ATPGD_eps=0.031373_steps=10_lambda=1_bn=frozen_run=v1")
        self.assertEqual(suffixes[1][1], dict(attack="PGD", eps=0.031373, steps=10, **{"lambda": "1"},
                                              bn_policy="frozen", run_tag="v1"))

    def test_header_mismatch_and_identity_mismatch_are_rejected(self):
        identity = dict(Scenario="S", seed=0, rate=1.0, model_name="M", epoch="best", model_seed=42,
                        ft_seed=0, strategy="FT-AL", ft_size=25000, training_size=25000, family="cnn",
                        subset_seed=42, group_seed=42, checkpoint="c", checkpoint_sha256="x",
                        plan_sha256="y", base_model_name="B", attack="PGD", eps=0.031373, steps=10,
                        **{"lambda": "1"}, bn_policy="frozen", run_tag="v1")
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.csv"
            with bad.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=BEST_CSV_COLUMNS).writeheader()
            with self.assertRaises(ValueError):
                mft._missing_ft_at_grid(bad, identity, [25000], [50])

            good = Path(d) / "good.csv"
            row = dict(identity, bins=50, in_size=25000, out_size=10000, in_size_rate="1",
                       timestamp="t")
            row.update({"I(X;T)-In": "1", "I(T;Y)-In": "2", "I(X;T)-Out": "3", "I(T;Y)-Out": "4"})
            mft._append_ft_at_row(good, row)
            missing, out = mft._missing_ft_at_grid(good, identity, [25000], [50, 100])
            self.assertEqual(missing, [(25000, 100)])
            self.assertIn(50, out)
            changed = dict(identity, checkpoint_sha256="different")
            with self.assertRaises(ValueError):
                mft._missing_ft_at_grid(good, changed, [25000], [50])


if __name__ == "__main__":
    unittest.main()
