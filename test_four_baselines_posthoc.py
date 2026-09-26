"""CPU tests for run_four_baselines_posthoc.py (python -m unittest test_four_baselines_posthoc -v)."""
import csv
import tempfile
import unittest
from pathlib import Path

import main_at_posthoc as mp
import run_four_baselines_posthoc as fb

TRAIN_PLAN = "saved_exp_plan/at_posthoc_plan/CIFAR100_RES18_PostAT_FTAL_DKD_Knockoff.yaml"


def plan_with(root_sets):
    return {"Victim": {}, "Baselines": {}, "Evaluation_Output": "x", "Case_Sets": root_sets}


class TestCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=fb.ROOT, prefix="_t")   # short root: Windows MAX_PATH
        self.models = Path(self.tmp.name) / "saved_models"
        _, runs = mp.expand_plan(fb.ROOT / TRAIN_PLAN, [0], "v1")
        self.names = [r[-1] for r in runs]
        # one finished run (epoch_-1..29 + best files), one unfinished run, the rest absent
        done = self.models / "at_posthoc" / self.names[0]
        done.mkdir(parents=True)
        for e in range(-1, 30):
            (done / f"epoch_{e}.pth").write_bytes(f"w{e}".encode())
        (done / "best_clean_epoch.pth").write_bytes(b"best")
        partial = self.models / "at_posthoc" / self.names[1]
        partial.mkdir(parents=True)
        (partial / "epoch_0.pth").write_bytes(b"p")
        src = self.models / "kd_final" / "CIFAR-100_ResNet-18to18_25000_DKD_0_0.0"
        src.mkdir(parents=True)
        (src / "best_epoch.pth").write_bytes(b"src")
        self.plan = plan_with({
            "sources": {"Stage": "source", "Checkpoint": "best_epoch.pth",
                        "Model_Path": ["kd_final/CIFAR-100_ResNet-18to18_25000_DKD_0_0.0", "kd_final/absent"]},
            "best_clean": {"Stage": "post_at", "Checkpoint": "best_clean_epoch.pth", "From_Training_Plan": TRAIN_PLAN},
            "trajectory": {"State": "all", "From_Training_Plan": TRAIN_PLAN},
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_ready_missing_and_metadata(self):
        cases, missing = fb.expand_cases(self.plan, ["sources", "best_clean"], models_root=self.models)
        self.assertEqual([c["Case_Set"] for c in cases], ["sources", "best_clean"])
        post = cases[1]
        self.assertEqual((post["Family"], post["AT_Eps"], post["AT_Seed"], post["Checkpoint"], post["Epoch"]),
                         ("FT-AL", 0.007843, 0, "best_clean_epoch", ""))
        self.assertEqual(cases[0]["Family"], "DKD")
        self.assertEqual(len(missing), 1 + 8)             # absent source + 8 unfinished/absent runs
        self.assertIn(("best_clean", f"at_posthoc/{self.names[1]}"), missing)   # partial run is not ready

    def test_trajectory_order_and_ids(self):
        cases, _ = fb.expand_cases(self.plan, ["trajectory"], models_root=self.models)
        self.assertEqual([c["Epoch"] for c in cases], list(range(-1, 30)))
        self.assertEqual(len({c["Case_ID"] for c in cases}), 31)
        cases, _ = fb.expand_cases(self.plan, ["trajectory"], select=["DKD"], models_root=self.models)
        self.assertEqual(cases, [])

    def test_merge_appends_and_refuses_changed_checkpoint(self):
        cases, _ = fb.expand_cases(self.plan, ["sources", "best_clean"], models_root=self.models)
        rows = fb.merge_rows([], cases[:1], root=self.models.parent)
        self.assertEqual(rows[0]["IPGuard_Status"], "pending")
        rows[0]["IPGuard_Status"] = "complete"
        rows = fb.merge_rows(rows, cases, root=self.models.parent)    # resume: keeps row 1, appends row 2
        self.assertEqual([r["IPGuard_Status"] for r in rows], ["complete", "pending"])
        (self.models / "kd_final/CIFAR-100_ResNet-18to18_25000_DKD_0_0.0/best_epoch.pth").write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            fb.merge_rows(rows, cases, root=self.models.parent)


class TestIPGuardTau(unittest.TestCase):
    def test_tau_is_max_mean_over_complete_overlap0_negatives(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.csv"
            vid = "CIFAR-100_25000_seed=42_overlap=1.0"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Scenario_Name", "Victim_Arch", "Suspect_Arch", "Suspect_Type", "FP_Config", "k_Param",
                            "Size", "Matching_Rate"])
                for seed, rates in [(42, [0.0, 0.02, 0.04, 0.06]), (43, [0.1, 0.1, 0.1, 0.1])]:
                    for tag, r in zip(fb.IPGUARD_TAGS, rates):
                        w.writerow([f"{vid}_seed={seed}_overlap=0.0", "ResNet-18", "ResNet-18", "negative", tag, 5, 100, r])
                for tag in fb.IPGUARD_TAGS:     # excluded: overlap 1.0, other k, incomplete model
                    w.writerow([f"{vid}_seed=44_overlap=1.0", "ResNet-18", "ResNet-18", "negative", tag, 5, 100, 0.9])
                    w.writerow([f"{vid}_seed=45_overlap=0.0", "ResNet-18", "ResNet-18", "negative", tag, 10, 100, 0.9])
                w.writerow([f"{vid}_seed=46_overlap=0.0", "ResNet-18", "ResNet-18", "negative", "TR", 5, 100, 0.9])
            assets = {"ipguard_negatives": path, "victim_id": vid, "arch": "ResNet-18"}
            tau, n = fb.ipguard_tau(assets, {"Baselines": {"IPGuard": {"k": 5, "Size": 100, "Decision": "Mean"}}})
            self.assertEqual(n, 2)                           # seed 46 lacks three configs
            self.assertAlmostEqual(tau, 0.1)
            tau, n = fb.ipguard_tau(assets, {"Baselines": {"IPGuard": {"k": 5, "Size": 100}}})   # default TR
            self.assertEqual(n, 3)                           # seed 46 has a TR row (0.9) -> counted
            self.assertAlmostEqual(tau, 0.9)
            with self.assertRaises(ValueError):
                fb.ipguard_tau(assets, {"Baselines": {"IPGuard": {"k": 5, "Size": 100, "Decision": "XX"}}})


if __name__ == "__main__":
    unittest.main()
