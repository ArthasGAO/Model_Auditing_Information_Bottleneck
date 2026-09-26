"""Verify DeiT second-row identities and unchanged first-row/style settings."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell

BASE = Path(__file__).resolve().parent


class DeiTGridTests(unittest.TestCase):
    def test_deit_row_and_scope(self):
        ns, book = notebook_namespace()
        old = json.loads((BASE/"tmp/distribution_check.before_deit_grid_20260921.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(book["cells"]), len(old["cells"]))
        for before, after in zip(old["cells"], book["cells"]):
            if after.get("id") != "fixed-h0-grid-render":
                self.assertEqual(before, after)
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config"):
            execute_cell(book, identity, ns)
        execute_cell(old, "fixed-h0-grid-render", ns)
        old_fig, old_axes = ns["fig_grid"], ns["axes_grid"]
        old_points = ns["grid_point_results"]
        execute_cell(book, "fixed-h0-grid-render", ns)
        fig, axes, points = ns["fig_grid"], ns["axes_grid"], ns["grid_point_results"]
        self.assertEqual(len(points), 600)
        np.testing.assert_array_equal(fig.get_size_inches(), old_fig.get_size_inches())
        self.assertEqual([t.get_text() for t in fig.texts], [t.get_text() for t in old_fig.texts])
        pd.testing.assert_frame_equal(points.query("panel <= 4"), old_points.query("panel <= 4"))
        for current, previous in zip(axes[0], old_axes[0]):
            self.assertEqual(len(current.collections), len(previous.collections))
            for c, p in zip(current.collections, previous.collections):
                np.testing.assert_array_equal(c.get_offsets(), p.get_offsets())
                np.testing.assert_array_equal(c.get_sizes(), p.get_sizes())
                np.testing.assert_array_equal(c.get_facecolors(), p.get_facecolors())
            np.testing.assert_array_equal(current.lines[0].get_xydata(), previous.lines[0].get_xydata())
            np.testing.assert_array_equal(current.get_xlim(), previous.get_xlim())
            np.testing.assert_array_equal(current.get_ylim(), previous.get_ylim())
        null = ns["fixed_h0"].load_nulls([ns["DEIT_NEGATIVE_SCENARIO"]])[ns["DEIT_NEGATIVE_SCENARIO"]]
        victim = ns["_resolve_victim_point"](ns["_deit_victim"])
        for index in range(4, 8):
            panel = index+1
            actual = points.query("panel == @panel and role == 'negative'")
            self.assertEqual(set(actual.model_name), set(null.evaluation.model_name))
            self.assertEqual(set(actual.h0), {"c10_deit"})
            ax = axes.flat[index]
            np.testing.assert_array_equal(ax.lines[0].get_xydata(), null.boundary())
            np.testing.assert_array_equal(ax.collections[1].get_offsets(), null.reference[["I(X;T)-In", "I(T;Y)-In"]])
            np.testing.assert_array_equal(ax.collections[-1].get_offsets(), [[victim["ix"], victim["iy"]]])
            if index < 7:
                self.assertTrue(points.query("panel == @panel and role == 'positive'").empty)
                self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), 81)
        positive = points.query("panel == 8 and role == 'positive'")
        self.assertEqual(sorted(positive.seed.tolist()), list(range(50)))
        csv_path = BASE/"saved_logs/extraction_final/MI_master_table_extraction_multiple.csv"
        self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                         "c4332c21d09af591e44d9770495331b3d90f036ce7c10e211e7d247fcb960f71")
        source = pd.read_csv(csv_path)
        source = source[(source.Scenario == ns["DEIT_EXTRACTION_SCENARIO"]) & (source.bins == 50) & (source.in_size == 25000)]
        self.assertEqual(len(source), 50)
        source = source.set_index("model_name").loc[positive.model_name]
        np.testing.assert_array_equal(positive[["ixt", "ity"]], source[["I(X;T)-In", "I(T;Y)-In"]])
        out = BASE/"tmp/fixed_h0_plot_validation/grid_deit.png"
        ns["fixed_h0"].save_paper_figure(fig, out, dpi=200)
        print(ns["grid_summary"][["panel", "group", "role", "n"]].to_string(index=False))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
