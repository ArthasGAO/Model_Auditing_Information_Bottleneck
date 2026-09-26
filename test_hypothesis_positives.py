"""Contract tests for the positive-suspect driver.

No Torch, no checkpoints: plans, MI tables and outputs are all temporary.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_hypothesis_test_fixed_splits as neg
import run_hypothesis_test_positives as pos


H0_SCENARIO = "CIFAR-10_ResNet-18_25000"
POS_SCENARIO = "CIFAR-10_ResNet-18_25000_PseudoLabel_25000"
MI_COLUMNS = ["Scenario", "seed", "rate", "model_name", "epoch", "bins", "in_size",
              "I(X;T)-In", "I(T;Y)-In", "strategy"]


def inventory():
    return {case: {str(s): {"model_name": f"{case}_{s}_0.0",
                            "relative_dir": f"CNN/{case}_{s}_0.0"}
                   for s in range(42, 122)} for case in neg.SCENARIOS}


class PositiveDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with patch.object(neg, "discover_models", return_value=inventory()):
            self.document = neg.generate_manifest(Path("unused"), 20260914)
        self.plan_dir = self.root / "plans"
        self.plan_dir.mkdir()
        self.write_plan("pos.yaml", {"Scenario_Name": POS_SCENARIO})
        self.mi_csv = self.root / "positives.csv"
        self.write_mi(self.default_rows())

    # ---- fixtures ----------------------------------------------------
    def write_plan(self, name, mapping):
        import yaml
        (self.plan_dir / name).write_text(yaml.safe_dump(mapping), encoding="utf-8")

    def default_rows(self):
        rows = []
        for i, strategy in enumerate(["FT-AL", "FT-LL"]):
            for seed in range(3):
                rows.append([POS_SCENARIO, seed, 1.0,
                             f"{POS_SCENARIO}_42_1.0_{strategy}_ftseed={seed}",
                             "best", 50, 25000, 9.0 + i, 3.0, strategy])
        # a row from a scenario the plan does not declare
        rows.append(["OTHER_SCENARIO", 0, 1.0, "other_model", "best", 50, 25000,
                     9.0, 3.0, "FT-AL"])
        return rows

    def write_mi(self, rows):
        with self.mi_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(MI_COLUMNS)
            writer.writerows(rows)

    def family(self, **overrides):
        base = {"label": "ft", "plan_dir": self.plan_dir, "mi_csv": self.mi_csv,
                "h0_scenario": H0_SCENARIO, "group_by": ("Scenario", "strategy")}
        base.update(overrides)
        return pos.normalize_family(base)

    # ---- plan parsing -------------------------------------------------
    def test_scenarios_from_plan_understands_both_plan_shapes(self):
        self.write_plan("at.yaml", {"Model_Path": [
            "/victim_models/CIFAR-10_ResNet-18_25000_42_1.0",
            "/extraction_vanilla/CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18_0_1.0"]})
        self.assertEqual(pos.scenarios_from_plan(self.plan_dir / "pos.yaml"),
                         [POS_SCENARIO])
        self.assertEqual(pos.scenarios_from_plan(self.plan_dir / "at.yaml"),
                         [H0_SCENARIO,
                          "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18"])
        # Scenario_Name wins when a plan carries both.
        self.write_plan("both.yaml", {"Scenario_Name": "X_1_2",
                                      "Model_Path": ["/a/Y_3_4"]})
        self.assertEqual(pos.scenarios_from_plan(self.plan_dir / "both.yaml"), ["X_1_2"])
        self.write_plan("bad.yaml", {"Optimizer": {}})
        with self.assertRaisesRegex(ValueError, "neither Scenario_Name nor Model_Path"):
            pos.scenarios_from_plan(self.plan_dir / "bad.yaml")
        self.write_plan("short.yaml", {"Model_Path": ["/a/NoSuffix"]})
        with self.assertRaisesRegex(ValueError, "no _<seed>_<rate> suffix"):
            pos.scenarios_from_plan(self.plan_dir / "short.yaml")

    def test_plan_dir_rejects_duplicate_scenarios_and_empty_dirs(self):
        self.write_plan("copy.yaml", {"Scenario_Name": POS_SCENARIO})
        with self.assertRaisesRegex(ValueError, "declare scenario"):
            pos.scenarios_from_plan_dir(self.plan_dir)
        (self.plan_dir / "copy.yaml").unlink()
        self.assertEqual(list(pos.scenarios_from_plan_dir(self.plan_dir)), [POS_SCENARIO])
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No \\*.yaml plans"):
            pos.scenarios_from_plan_dir(empty)
        with self.assertRaisesRegex(ValueError, "Plan directory not found"):
            pos.scenarios_from_plan_dir(self.root / "absent")

    def test_normalize_family_defaults_and_rejections(self):
        f = pos.normalize_family({"label": "x", "plan_dir": self.plan_dir,
                                  "mi_csv": self.mi_csv, "h0_scenario": H0_SCENARIO})
        self.assertEqual(f["group_by"], ("Scenario",))
        f = pos.normalize_family({"label": "x", "plan_dir": self.plan_dir,
                                  "mi_csv": self.mi_csv, "h0_scenario": H0_SCENARIO,
                                  "group_by": ("strategy",)})
        self.assertEqual(f["group_by"], ("Scenario", "strategy"))   # always prepended
        with self.assertRaisesRegex(ValueError, "missing keys"):
            pos.normalize_family({"label": "x"})
        with self.assertRaisesRegex(ValueError, "duplicates"):
            pos.normalize_family({"label": "x", "plan_dir": self.plan_dir,
                                  "mi_csv": self.mi_csv, "h0_scenario": H0_SCENARIO,
                                  "group_by": ("strategy", "strategy")})

    # ---- positive selection -------------------------------------------
    def test_selection_is_plan_based_and_filterable(self):
        groups, values, _, report = pos.preflight_positives(
            self.family(), 25000, 50, "In")
        self.assertEqual(len(values), 6)                       # the stray row is excluded
        self.assertNotIn("other_model", values)
        self.assertEqual(report["rows_outside_declared_scenarios"], 1)
        self.assertEqual(report["scenarios_without_rows"], [])
        self.assertEqual(sorted(groups), [(POS_SCENARIO, "FT-AL"), (POS_SCENARIO, "FT-LL")])
        self.assertEqual(len(groups[(POS_SCENARIO, "FT-AL")]), 3)

        groups, values, _, report = pos.preflight_positives(
            self.family(filters={"strategy": ["FT-LL"]}), 25000, 50, "In")
        self.assertEqual(len(values), 3)
        self.assertEqual(report["rows_removed_by_filters"], 3)
        self.assertEqual(list(groups), [(POS_SCENARIO, "FT-LL")])

        # a declared scenario with no rows is reported, not silently dropped
        self.write_plan("missing.yaml", {"Scenario_Name": "NOT_IN_TABLE"})
        _, _, _, report = pos.preflight_positives(self.family(), 25000, 50, "In")
        self.assertEqual(report["scenarios_without_rows"], ["NOT_IN_TABLE"])

    def test_selection_rejects_bad_tables_and_explains_empty_results(self):
        with self.assertRaisesRegex(ValueError, "no positive rows"):
            pos.preflight_positives(self.family(), 12500, 50, "In")   # no such in_size
        with self.assertRaisesRegex(ValueError, "filter column"):
            pos.preflight_positives(self.family(filters={"nope": ["x"]}), 25000, 50, "In")
        with self.assertRaisesRegex(ValueError, "missing from"):
            pos.preflight_positives(self.family(group_by=("Scenario", "nope")),
                                    25000, 50, "In")
        rows = self.default_rows()
        self.write_mi(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate MI row"):
            pos.preflight_positives(self.family(), 25000, 50, "In")
        rows = self.default_rows()
        rows[0][7] = "nan"
        self.write_mi(rows)
        with self.assertRaisesRegex(ValueError, "nonfinite MI"):
            pos.preflight_positives(self.family(), 25000, 50, "In")

    # ---- evaluation ----------------------------------------------------
    def h0_values(self, rng):
        return {m["model_name"]: rng.normal(size=2).tolist()
                for case in self.document["payload"]["cases"].values()
                for m in case.values()}

    def run_eval(self, family, h0_values, out, **selection):
        resolved = dict(family)
        groups, values, family_hash, report = pos.preflight_positives(
            family, 25000, 50, "In")
        resolved.update({
            "_groups": groups, "_values": values, "_hash": family_hash,
            "_report": report,
            "_group_of": {n: "|".join(k) for k, ms in groups.items() for n in ms},
            "_scenario_of": {n: k[0] for k, ms in groups.items() for n in ms},
        })
        return pos.evaluate_positives(
            self.document, [resolved], h0_values, "h0hash", output_dir=out,
            in_size={c: 25000 for c in self.document["payload"]["cases"]},
            bins=50, mi_kind="In", alphas=[0.05], in_size_rate=1.0,
            training_sizes={c: 25000 for c in self.document["payload"]["cases"]}, **selection)

    def test_selected_round_keeps_manifest_ids_and_single_round_sd_is_missing(self):
        h0 = self.h0_values(np.random.default_rng(5))
        out = self.root / "selected"
        summary = self.run_eval(self.family(), h0, out, round_ids=[44], k_values=[30])
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(r["n_rounds"] == 1 and r["k_ref"] == 30 for r in summary))
        self.assertTrue(all(r["std_tpr_F@0.05"] is None for r in summary))
        with (out/"per_model.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["round_id"] for r in rows}, {"44"})
        meta = json.loads((out/"run_metadata.json").read_text())
        self.assertEqual(meta["round_ids"], [44])
        self.assertEqual(meta["k_values"], [30])
        ref = [h0[m["model_name"]] for m in neg.get_split(self.document, H0_SCENARIO, 44, 30)["h0"]]
        xy = [[float(r["ixt"]), float(r["ity"])] for r in rows]
        np.testing.assert_allclose([float(r["p_F"]) for r in rows], neg.score(ref, xy)[4], rtol=1e-14)
        for selection in ({"round_ids": []}, {"round_ids": [50]}, {"round_ids": [44, 44]},
                          {"k_values": [29]}, {"k_values": [30, 30]}):
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                self.run_eval(self.family(), h0, self.root/"bad_selection", **selection)
        self.assertFalse((self.root/"bad_selection").exists())

    def test_distinct_selected_round_per_k(self):
        h0 = self.h0_values(np.random.default_rng(19))
        out = self.root / "per_k"
        summary = self.run_eval(
            self.family(), h0, out, k_values=[5, 30],
            round_by_h0_k={(H0_SCENARIO, 5): 2, (H0_SCENARIO, 30): 44})
        self.assertEqual(len(summary), 4)  # two positive groups x two k values
        self.assertTrue(all(row["n_rounds"] == 1 and row["std_tpr_F@0.05"] is None
                            for row in summary))
        with (out / "per_split.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({(int(row["k_ref"]), int(row["round_id"])) for row in rows},
                         {(5, 2), (30, 44)})
        metadata = json.loads((out / "run_metadata.json").read_text())
        self.assertEqual(metadata["selected_round_by_h0_k"],
                         {H0_SCENARIO: {"5": 2, "30": 44}})

    def test_reference_matches_get_split_and_tpr_is_correct(self):
        rng = np.random.default_rng(5)
        h0 = self.h0_values(rng)
        out = self.root / "run"
        summary = self.run_eval(self.family(), h0, out)
        self.assertEqual(len(summary), 2 * len(neg.K_VALUES))     # 2 groups x 6 k

        with (out / "per_split.csv").open(newline="", encoding="utf-8") as stream:
            splits = list(csv.DictReader(stream))
        self.assertEqual(len(splits), 2 * len(neg.K_VALUES) * 50)

        # Every recorded reference must be the manifest's h0 for that (round, k),
        # and mu / Sigma must equal a direct refit of those same models.
        for row in splits[:40]:
            r, k = int(row["round_id"]), int(row["k_ref"])
            expected = [m["seed"] for m in
                        neg.get_split(self.document, H0_SCENARIO, r, k)["h0"]]
            self.assertEqual(json.loads(row["h0_seeds"]), expected)
            ref = np.array([h0[m["model_name"]] for m in
                            neg.get_split(self.document, H0_SCENARIO, r, k)["h0"]])
            np.testing.assert_allclose(json.loads(row["mu"]), ref.mean(axis=0))
            np.testing.assert_allclose(json.loads(row["covariance"]),
                                       np.cov(ref, rowvar=False, ddof=1))

        # TPR must equal the fraction of that group's positives with p < alpha.
        with (out / "per_model.csv").open(newline="", encoding="utf-8") as stream:
            per_model = list(csv.DictReader(stream))
        self.assertEqual(len(per_model), 6 * len(neg.K_VALUES) * 50)
        for row in splits[:20]:
            members = [m for m in per_model
                       if m["round_id"] == row["round_id"]
                       and m["k_ref"] == row["k_ref"] and m["group"] == row["group"]]
            self.assertEqual(len(members), int(row["n_positive"]))
            hits = sum(1 for m in members if float(m["p_F"]) < 0.05)
            self.assertEqual(hits, int(row["ntp_F@0.05"]))
            self.assertAlmostEqual(hits / len(members), float(row["tpr_F@0.05"]))

        # Each law's statistic is written next to its own p-value and must
        # reproduce it; chi2 takes the raw T2, F takes the scaled one.
        from scipy.stats import chi2 as chi2_dist, f as f_dist
        for row in per_model[:200]:
            k = int(row["k_ref"])
            t2, s_chi2, s_f = (float(row["T2"]), float(row["stat_chi2"]),
                               float(row["stat_F"]))
            self.assertEqual(s_chi2, t2)
            self.assertAlmostEqual(s_f, t2 * (k - 2) / (2 * (k - 1)), places=12)
            self.assertAlmostEqual(chi2_dist.sf(s_chi2, 2), float(row["p_chi2"]), places=12)
            self.assertAlmostEqual(f_dist.sf(s_f, 2, k - 2), float(row["p_F"]), places=12)

        # summary.csv carries each law's p-value spread for the cell.
        with (out / "summary.csv").open(newline="", encoding="utf-8") as stream:
            summary_rows = list(csv.DictReader(stream))
        for row in summary_rows:
            cell = [m for m in per_model
                    if m["k_ref"] == row["k_ref"] and m["group"] == row["group"]]
            self.assertEqual(int(row["n_eval_total"]), len(cell))
            for law, column in [("chi2", "p_chi2"), ("F", "p_F")]:
                values = np.array([float(m[column]) for m in cell])
                self.assertAlmostEqual(float(row[f"p_{law}_min"]), values.min())
                self.assertAlmostEqual(float(row[f"p_{law}_max"]), values.max())
                self.assertAlmostEqual(float(row[f"p_{law}_median"]), float(np.median(values)))
                self.assertAlmostEqual(float(row[f"p_{law}_mean"]), float(values.mean()))
                self.assertAlmostEqual(float(row[f"p_{law}_p05"]),
                                       float(np.percentile(values, 5)))
                self.assertAlmostEqual(float(row[f"p_{law}_p95"]),
                                       float(np.percentile(values, 95)))
            # The recorded TPR must equal the share of that cell below alpha.
            share = float(np.mean([float(m["p_F"]) < 0.05 for m in cell]))
            self.assertAlmostEqual(share, float(row["mean_tpr_F@0.05"]))
        self.assertFalse([c for c in summary_rows[0]
                          if c.startswith(("crit_", "T2_", "f_scale"))])

        # The suspect-side KS diagnostics are dropped; positives are not null draws.
        self.assertFalse([c for c in splits[0] if c.startswith("susp_")])
        self.assertIn("mardia_skew_p", splits[0])
        meta = json.loads((out / "run_metadata.json").read_text())
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(meta["families"][0]["n_positive"], 6)

    def test_separated_positives_are_detected_and_overlapping_ones_are_not(self):
        rng = np.random.default_rng(11)
        h0 = {name: v for name, v in self.h0_values(rng).items()}
        out = self.root / "run_sep"
        # FT-AL sits at (9, 3), far from the standard-normal reference; FT-LL at
        # (10, 3) is even further. Both must be rejected in every round.
        summary = self.run_eval(self.family(), h0, out)
        for row in summary:
            self.assertEqual(float(row["mean_tpr_F@0.05"]), 1.0)

        # Now place the positives inside the reference cloud: no detection.
        rows = self.default_rows()
        for row in rows[:6]:
            row[7], row[8] = 0.0, 0.0
        self.write_mi(rows)
        out = self.root / "run_overlap"
        summary = self.run_eval(self.family(), h0, out)
        self.assertLess(max(float(r["mean_tpr_F@0.05"]) for r in summary), 0.5)

    def test_output_directory_is_never_reused(self):
        rng = np.random.default_rng(3)
        h0 = self.h0_values(rng)
        out = self.root / "once"
        self.run_eval(self.family(), h0, out)
        with self.assertRaises(FileExistsError):
            self.run_eval(self.family(), h0, out)


if __name__ == "__main__":
    unittest.main()
