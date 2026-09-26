"""Contract tests for saved split identity and statistical compatibility."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import f
import run_hypothesis_test_fixed_splits as experiment


def inventory():
    return {case: {str(s): {"model_name": f"{case}_{s}_0.0",
                            "relative_dir": f"CNN/{case}_{s}_0.0"}
                   for s in range(42, 122)} for case in experiment.SCENARIOS}


class FixedSplitsTests(unittest.TestCase):
    def test_rate_grid_uses_case_denominator_without_rounding(self):
        grid = experiment.rate_configuration_grid(
            [.05, .10, .20, .50, .75, 1.0], [30, 50], {"a": 25000, "b": 10000}, ["a", "b"])
        self.assertEqual(len(grid), 12)
        self.assertEqual([sizes["a"] for _, sizes, b in grid if b == 50],
                         [1250, 2500, 5000, 12500, 18750, 25000])
        self.assertEqual([sizes["b"] for _, sizes, b in grid if b == 50],
                         [500, 1000, 2000, 5000, 7500, 10000])
        for rates in [[], [5], [0], [True], [.1, .1], [float("nan")]]:
            with self.assertRaises(ValueError):
                experiment.rate_configuration_grid(rates, [50], {"a": 25000}, ["a"])
        with self.assertRaisesRegex(ValueError, "no silent rounding"):
            experiment.rate_configuration_grid([.05], [50], {"a": 21}, ["a"])

    def setUp(self):
        with patch.object(experiment, "discover_models", return_value=inventory()):
            self.document = experiment.generate_manifest(Path("unused"), 20260914)

    def test_roundtrip_pairing_and_reproducibility(self):
        with patch.object(experiment, "discover_models", return_value=inventory()):
            again = experiment.generate_manifest(Path("unused"), 20260914)
        self.assertEqual(self.document, again)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            experiment.write_json(path, self.document)
            with patch.object(experiment.random, "Random", side_effect=AssertionError("No resampling")):
                loaded = experiment.load_manifest(path)
                for r in range(50):
                    first = experiment.get_split(loaded, experiment.SCENARIOS[0], r, 5)
                    for k in experiment.K_VALUES:
                        split = experiment.get_split(loaded, experiment.SCENARIOS[0], r, k)
                        self.assertEqual(first["evaluation_negative"], split["evaluation_negative"])
                        self.assertEqual(first["h0"], split["h0"][:5])
                        self.assertEqual(len(split["h0"]), k)
                        self.assertEqual(len(split["unused"]), 30 - k)
                        self.assertFalse({m["seed"] for m in split["h0"]} &
                                         {m["seed"] for m in split["evaluation_negative"]})
            with self.assertRaises(FileExistsError):
                experiment.write_json(path, self.document)

    def test_corruption_and_overlap_rejected(self):
        changed = copy.deepcopy(self.document)
        changed["payload"]["rounds"][0]["eval_negative_seeds"][0] = -1
        with self.assertRaisesRegex(ValueError, "hash"):
            experiment.validate_manifest(changed)
        changed["manifest_sha256"] = experiment.payload_hash(changed["payload"])
        with self.assertRaisesRegex(ValueError, "Overlap/invalid"):
            experiment.validate_manifest(changed)

    def test_csv_duplicate_missing_and_nonfinite_rejected(self):
        header = ["model_name", "Scenario", "seed", "rate", "in_size", "bins", "I(X;T)-In", "I(T;Y)-In"]
        rows = [[m["model_name"], case, seed, 0, 25000, 50, 1.2, 0.8]
                for case, models in inventory().items() for seed, m in models.items()]
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, "discover_models", return_value=inventory()):
            path = Path(directory) / "mi.csv"
            def check(data):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(data)
                return experiment.preflight(self.document, Path(directory), path, 25000, 50, "In")
            self.assertEqual(len(check(rows)[0]), 480)
            # Each scenario may have a different actual MI size at the same rate.
            per_case = {case: 10000 if i % 2 else 25000
                        for i, case in enumerate(experiment.SCENARIOS)}
            varied = copy.deepcopy(rows)
            for row in varied:
                row[4] = per_case[row[1]]
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(varied)
            values, _ = experiment.preflight(self.document, Path(directory), path,
                                             per_case, 50, "In")
            self.assertEqual(len(values), 480)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                check(rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "Missing MI"):
                check(rows[1:])
            invalid = copy.deepcopy(rows)
            invalid[0][-1] = "nan"
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                check(invalid)

    def test_f_formula_and_legacy_compatibility(self):
        import run_hypothesis_test as legacy
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(30, 2)), rng.normal(size=(50, 2))
        for k in experiment.K_VALUES:
            mu, cov, t2, pc, pf = experiment.score(x[:k], y)
            pool = {0.0: [{"seed": i, "mi": value} for i, value in enumerate(y)]}
            old = legacy.score_pool0_suspects(pool, mu, cov, k)
            np.testing.assert_allclose(t2, [r["T2"] for r in old], rtol=1e-12)
            np.testing.assert_allclose(pc, [r["p_chi2"] for r in old], rtol=1e-12)
            np.testing.assert_allclose(pf, [r["p_F"] for r in old], rtol=1e-12)
            self.assertAlmostEqual(float(f.sf(f.isf(.05, 2, k-2), 2, k-2)), .05)
            diag = experiment.diagnostics(x[:k], t2)
            legacy_diag = legacy.reference_diagnostics(x[:k])
            self.assertAlmostEqual(diag["mardia_skew_p"], legacy_diag["mardia"]["skew_p"])
            self.assertAlmostEqual(diag["loo_ks_p_F_descriptive"], legacy_diag["loo_ks_p_F"])

    def test_law_statistics_pair_each_p_value_with_its_own_input(self):
        from scipy.stats import chi2
        rng = np.random.default_rng(13)
        for k in experiment.K_VALUES:
            x, y = rng.normal(size=(k, 2)), rng.normal(size=(20, 2))
            _, _, t2, p_chi2, p_f = experiment.score(x, y)
            stat_chi2, stat_f = experiment.law_statistics(t2, k)
            # chi2 is fed the raw T2; F is fed the scaled one.
            np.testing.assert_array_equal(stat_chi2, t2)
            np.testing.assert_allclose(stat_f, t2 * (k - 2) / (2 * (k - 1)), rtol=1e-15)
            # Each recorded statistic must reproduce its own recorded p-value.
            np.testing.assert_allclose(chi2.sf(stat_chi2, 2), p_chi2, rtol=1e-12)
            np.testing.assert_allclose(f.sf(stat_f, 2, k - 2), p_f, rtol=1e-12)
            self.assertTrue((stat_f < stat_chi2).all())   # the factor is below 1 for p=2

    def test_summary_statistics_spread_per_law(self):
        rng = np.random.default_rng(29)
        chi2_blocks = [rng.uniform(size=50) for _ in range(50)]
        f_blocks = [rng.uniform(size=50) for _ in range(50)]
        out = experiment.summary_statistics({"chi2": chi2_blocks, "F": f_blocks})
        self.assertEqual(out["n_eval_total"], 2500)
        for law, blocks in [("chi2", chi2_blocks), ("F", f_blocks)]:
            flat = np.concatenate(blocks)
            for part, expected in [("min", flat.min()), ("max", flat.max()),
                                   ("median", np.median(flat)),
                                   ("mean", flat.mean()),
                                   ("p05", np.percentile(flat, 5)),
                                   ("p95", np.percentile(flat, 95))]:
                self.assertAlmostEqual(out[f"p_{law}_{part}"], float(expected))
            self.assertLessEqual(out[f"p_{law}_min"], out[f"p_{law}_p05"])
            self.assertLessEqual(out[f"p_{law}_p05"], out[f"p_{law}_median"])
            self.assertLessEqual(out[f"p_{law}_median"], out[f"p_{law}_p95"])
            self.assertLessEqual(out[f"p_{law}_p95"], out[f"p_{law}_max"])
        # No threshold columns: p < alpha needs no k-dependent critical value.
        self.assertFalse([c for c in out if c.startswith(("crit_", "T2_", "f_scale"))])
        with self.assertRaisesRegex(ValueError, "at least one chi2 p-value"):
            experiment.summary_statistics({"chi2": [], "F": f_blocks})
        with self.assertRaisesRegex(ValueError, "same evaluations"):
            experiment.summary_statistics({"chi2": chi2_blocks,
                                           "F": [rng.uniform(size=3)]})

    def test_summary_spread_matches_the_rejection_rate(self):
        """The fraction below alpha must agree with the recorded quantiles."""
        rng = np.random.default_rng(31)
        blocks = [rng.uniform(size=100) for _ in range(20)]
        out = experiment.summary_statistics({"chi2": blocks, "F": blocks})
        flat = np.concatenate(blocks)
        # p05 is by definition the 5% point, so exactly 5% lie below it.
        self.assertAlmostEqual(float(np.mean(flat < out["p_F_p05"])), 0.05, places=2)
        self.assertAlmostEqual(float(np.mean(flat < out["p_F_median"])), 0.5, places=2)

    def test_singular_reference_fails(self):
        with self.assertRaisesRegex(ValueError, "Singular"):
            experiment.score(np.ones((5, 2)), np.zeros((50, 2)))

    def test_evaluate_one_case_preserves_full_manifest(self):
        import json
        rng = np.random.default_rng(901)
        case = experiment.SCENARIOS[1]
        values = {m["model_name"]: rng.normal(size=2).tolist()
                  for m in self.document["payload"]["cases"][case].values()}
        original = copy.deepcopy(self.document)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "single"
            with patch.object(experiment, "diagnostics", return_value={}):
                result = experiment.evaluate(
                    self.document, values, "input-hash", output_dir=out,
                    csv_path="input.csv", model_root="unused", in_size=25000,
                    bins=50, mi_kind="In", alphas=[.05, .01], scenarios=[case])
            self.assertEqual(len(result), 6)
            self.assertEqual({r["scenario"] for r in result}, {case})
            self.assertEqual(json.loads((out / "manifest.json").read_text()), original)
            meta = json.loads((out / "run_metadata.json").read_text())
            self.assertEqual(meta["completed_splits"], 300)
            self.assertEqual(meta["evaluated_scenarios"], [case])
            with (out / "per_model.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 15000)
            self.assertEqual({r["scenario"] for r in rows}, {case})
            self.assertEqual(len({(r["round_id"], r["k_ref"], r["seed"])
                                  for r in rows}), 15000)
            with self.assertRaisesRegex(ValueError, "Invalid evaluation scenarios"):
                experiment.evaluate(
                    self.document, values, "hash", output_dir=Path(directory)/"bad",
                    csv_path="input.csv", model_root="unused", in_size=25000,
                    bins=50, mi_kind="In", alphas=[.05], scenarios=["unknown"])
            self.assertFalse((Path(directory)/"bad").exists())
        self.assertEqual(self.document, original)

    def test_evaluate_distinct_selected_round_per_k(self):
        import json
        case = experiment.SCENARIOS[1]
        rng = np.random.default_rng(912)
        values = {m["model_name"]: rng.normal(size=2).tolist()
                  for m in self.document["payload"]["cases"][case].values()}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "selected"
            with patch.object(experiment, "diagnostics", return_value={}):
                result = experiment.evaluate(
                    self.document, values, "input-hash", output_dir=out,
                    csv_path="input.csv", model_root="unused", in_size=25000,
                    bins=50, mi_kind="In", alphas=[.01], scenarios=[case],
                    k_values=[5, 30], round_by_scenario_k={(case, 5): 2, (case, 30): 44})
            self.assertEqual(len(result), 2)
            self.assertTrue(all(row["n_rounds"] == 1 and row["std_fpr_F@0.01"] is None
                                for row in result))
            with (out / "per_split.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({(int(row["k_ref"]), int(row["round_id"])) for row in rows},
                             {(5, 2), (30, 44)})
            metadata = json.loads((out / "run_metadata.json").read_text())
            self.assertEqual(metadata["selected_round_by_scenario_k"],
                             {case: {"5": 2, "30": 44}})

    def test_victim_preflight_exactly_one_row_per_case(self):
        header = ["Scenario", "seed", "rate", "model_name", "in_size", "bins", "I(X;T)-In", "I(T;Y)-In"]
        rows = [[case, 42, 1.0, f"{case}_42_1.0", 2500, 50, 3.33, 3.32] for case in experiment.SCENARIOS]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "victim.csv"
            def check(data, size=2500):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(data)
                return experiment.preflight_victim(self.document, path, size, 50, "In")
            victims, _ = check(rows)
            self.assertEqual(set(victims), set(experiment.SCENARIOS))
            self.assertEqual(victims[experiment.SCENARIOS[0]], [3.33, 3.32])
            with self.assertRaisesRegex(ValueError, "Duplicate victim"):
                check(rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "Missing victim"):
                check(rows[1:])
            with self.assertRaisesRegex(ValueError, "Missing victim"):
                check(rows, size=25000)               # no fallback to another in_size
            other_seed = copy.deepcopy(rows)
            other_seed[0][1], other_seed[0][3] = 43, f"{rows[0][0]}_43_1.0"
            with self.assertRaisesRegex(ValueError, "Missing victim"):
                check(other_seed)                     # seed 43 is not the victim
            invalid = copy.deepcopy(rows)
            invalid[0][-1] = "inf"
            with self.assertRaisesRegex(ValueError, "Nonfinite victim"):
                check(invalid)
            renamed = copy.deepcopy(rows)
            renamed[0][3] = "wrong_name"
            with self.assertRaisesRegex(ValueError, "model_name mismatch"):
                check(renamed)

    def test_gate1_short_circuit_and_equality_passes(self):
        victim = [3.3, 3.32]
        self.assertTrue(experiment.passes_gate1([3.3, 3.32], victim))    # exact copy passes
        self.assertTrue(experiment.passes_gate1([4.6, 3.10], victim))    # lower-right passes
        self.assertFalse(experiment.passes_gate1([3.2, 3.10], victim))   # less I(X;T) fails
        self.assertFalse(experiment.passes_gate1([4.6, 3.33], victim))   # more I(T;Y) fails
        evaluation = np.array([[4.6, 3.10], [3.2, 3.10], [3.3, 3.32]])
        t2, pc, pf = np.array([5.0, 9.0, 0.5]), np.array([.08, .01, .78]), np.array([.1, .02, .8])
        g_t2, g_pc, g_pf, flags = experiment.apply_gate1(evaluation, victim, t2, pc, pf)
        self.assertEqual(flags, [True, False, True])
        np.testing.assert_array_equal(g_t2, [5.0, 0.0, 0.5])
        np.testing.assert_array_equal(g_pc, [.08, 1.0, .78])
        np.testing.assert_array_equal(g_pf, [.1, 1.0, .8])
        np.testing.assert_array_equal(t2, [5.0, 9.0, 0.5])              # inputs untouched


if __name__ == "__main__":
    unittest.main()
