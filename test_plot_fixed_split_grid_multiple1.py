"""Verify illustrative panel input coordinates and preserve all unaffected panels."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell
from synthetic_mi_repro.deit_multiple1.generate import verify

ROOT = Path(__file__).resolve().parent


class Multiple1GridTests(unittest.TestCase):
    def test_scope_and_coordinates(self):
        verify()
        ns, book = notebook_namespace()
        old = json.loads((ROOT / "tmp/distribution_check.before_deit_multiple1.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(book["cells"]), len(old["cells"]))
        for before, after in zip(old["cells"], book["cells"]):
            if after.get("id") != "fixed-h0-grid-render":
                self.assertEqual(before, after)
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config"):
            execute_cell(book, identity, ns)
        execute_cell(old, "fixed-h0-grid-render", ns)
        old_fig, old_axes, old_points = ns["fig_grid"], ns["axes_grid"], ns["grid_point_results"]
        execute_cell(book, "fixed-h0-grid-render", ns)
        fig, axes, points = ns["fig_grid"], ns["axes_grid"], ns["grid_point_results"]
        self.assertEqual(len(points), 700)
        np.testing.assert_array_equal(fig.get_size_inches(), old_fig.get_size_inches())
        self.assertEqual([t.get_text() for t in fig.texts], [t.get_text() for t in old_fig.texts])
        self.assertEqual([t.get_text() for t in fig.legends[0].texts],
                         [t.get_text() for t in old_fig.legends[0].texts])
        for i in range(8):
            ax, before = axes.flat[i], old_axes.flat[i]
            # Every boundary, negative cloud, reference cloud and victim is unchanged.
            np.testing.assert_array_equal(ax.lines[0].get_xydata(), before.lines[0].get_xydata())
            for j in (0, 1, -1):
                np.testing.assert_array_equal(ax.collections[j].get_offsets(), before.collections[j].get_offsets())
                np.testing.assert_array_equal(ax.collections[j].get_sizes(), before.collections[j].get_sizes())
                np.testing.assert_array_equal(ax.collections[j].get_facecolors(), before.collections[j].get_facecolors())
            if i not in (4, 5):
                panel = i + 1
                pd.testing.assert_frame_equal(points.query("panel == @panel").reset_index(drop=True),
                                              old_points.query("panel == @panel").reset_index(drop=True))
                self.assertEqual(len(ax.collections), len(before.collections))
                for actual, original in zip(ax.collections, before.collections):
                    np.testing.assert_array_equal(actual.get_offsets(), original.get_offsets())
                np.testing.assert_array_equal(ax.get_xlim(), before.get_xlim())
                np.testing.assert_array_equal(ax.get_ylim(), before.get_ylim())
        for i, csv_name in ((4, "CSV_DEIT_FT_MULTIPLE1"), (5, "CSV_DEIT_PRUNE_MULTIPLE1")):
            panel = i + 1
            positive = points.query("panel == @panel and role == 'positive'")
            self.assertEqual(sorted(positive.seed), list(range(50)))
            self.assertEqual(set(positive.h0), {"c10_deit"})
            source = pd.read_csv(ns[csv_name]).set_index("model_name").loc[positive.model_name]
            self.assertEqual(len(source), 50)
            np.testing.assert_array_equal(positive[["ixt", "ity"]], source[["I(X;T)-In", "I(T;Y)-In"]])
            np.testing.assert_array_equal(axes.flat[i].collections[2].get_facecolors(), axes.flat[0].collections[2].get_facecolors())
            self.assertTrue(source["I(X;T)-Out"].isna().all())
            self.assertTrue(source["checkpoint"].isna().all())
        self.assertTrue(points.query("panel in [3, 7] and role == 'positive'").empty)
        plan = json.loads((ROOT / "synthetic_mi_repro/deit_multiple1/plan.json").read_text(encoding="utf-8"))
        for role in ("source", "target"):
            scenario = plan["anchors"][role]["scenario"]
            null = ns["fixed_h0"].load_nulls([scenario])[scenario]
            self.assertEqual(set(null.evaluation.model_name),
                             {r["model_name"] for r in plan["anchors"][role]["evaluation_rows"]})
            np.testing.assert_allclose(null.evaluation[["I(X;T)-In", "I(T;Y)-In"]].mean(),
                                       plan["anchors"][role]["negative_mean"], rtol=0, atol=1e-14)
        output = ROOT / "tmp/fixed_h0_plot_validation/grid_deit_multiple1.png"
        ns["fixed_h0"].save_paper_figure(fig, output, dpi=200)
        print(ns["grid_summary"][["panel", "group", "role", "n"]].to_string(index=False))
        print(f"Preview: {output}")
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
