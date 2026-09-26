"""Check the bottom notebook cell changes only requested panel data."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from test_plot_fixed_split_h0 import notebook_namespace

BASE = Path(__file__).resolve().parent


def execute_cell(book, identity, namespace):
    source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == identity)
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    with patch("matplotlib.figure.Figure.savefig"), patch("matplotlib.pyplot.show"):
        exec(compile(source, identity, "exec"), namespace)


class GridDataTests(unittest.TestCase):
    def test_requested_panels_and_scope(self):
        ns, book = notebook_namespace()
        old = json.loads((BASE/"tmp/distribution_check.before_ft_prune_grid_20260921.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(book["cells"]), len(old["cells"]))
        for before, after in zip(old["cells"], book["cells"]):
            if after.get("id") != "fixed-h0-grid-render":
                self.assertEqual(before, after)
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config"):
            execute_cell(book, identity, ns)
        execute_cell(old, "fixed-h0-grid-render", ns)
        old_fig, old_axes = ns["fig_grid"], ns["axes_grid"]
        old_summary, old_points = ns["grid_summary"], ns["grid_point_results"]
        execute_cell(book, "fixed-h0-grid-render", ns)
        fig, axes = ns["fig_grid"], ns["axes_grid"]
        summary, points = ns["grid_summary"], ns["grid_point_results"]
        np.testing.assert_array_equal(fig.get_size_inches(), old_fig.get_size_inches())
        self.assertEqual([t.get_text() for t in fig.texts], [t.get_text() for t in old_fig.texts])
        self.assertEqual(len(points), 700)
        for index in (2, 6):
            panel = index+1
            self.assertEqual(len(points.query("panel == @panel and role == 'positive'")), 0)
            self.assertEqual(sum(len(c.get_offsets()) for c in axes.flat[index].collections), 81)
            self.assertEqual(ns["PLOT_PANELS"][index]["positive_groups"], [])
        for index in (0, 1):
            panel = index+1
            positive = points.query("panel == @panel and role == 'positive'")
            self.assertEqual(sorted(positive.seed.tolist()), list(range(50)))
            filename = ns["CSV_FT_MULTIPLE"] if index == 0 else ns["CSV_PRUNE_MULTIPLE"]
            source = pd.read_csv(filename)
            source = source[(source.Scenario == ns["RN18_FT_SCENARIO"]) &
                            (source.bins == ns["BINS"]) & (source.in_size == ns["IN_SIZE"]) &
                            (source.strategy == "FT-AL") & (source.epoch == "best")]
            if index == 1:
                source = source[(source.sparsity == .8) & (source.ckpt_kind == "best")]
            self.assertEqual(len(source), 50)
            source = source.set_index("model_name").loc[positive.model_name]
            np.testing.assert_array_equal(positive[["ixt", "ity"]].to_numpy(), source[["I(X;T)-In", "I(T;Y)-In"]].to_numpy())
        # Both fourth-column panels and the lower first/second panels unchanged.
        for index in (3, 4, 5, 7):
            for current, previous in zip(axes.flat[index].collections, old_axes.flat[index].collections):
                np.testing.assert_array_equal(current.get_offsets(), previous.get_offsets())
                np.testing.assert_array_equal(current.get_sizes(), previous.get_sizes())
                np.testing.assert_array_equal(current.get_facecolors(), previous.get_facecolors())
            np.testing.assert_array_equal(axes.flat[index].get_xlim(), old_axes.flat[index].get_xlim())
            np.testing.assert_array_equal(axes.flat[index].get_ylim(), old_axes.flat[index].get_ylim())
        pd.testing.assert_frame_equal(points.query("role == 'negative'").reset_index(drop=True),
                                      old_points.query("role == 'negative'").reset_index(drop=True))
        for current, previous in zip(axes.flat, old_axes.flat):
            np.testing.assert_array_equal(current.lines[0].get_xydata(), previous.lines[0].get_xydata())
            np.testing.assert_array_equal(current.collections[-1].get_offsets(), previous.collections[-1].get_offsets())
            self.assertEqual(len(current.child_axes), len(previous.child_axes))
        out = BASE/"tmp/fixed_h0_plot_validation/grid_ft_prune.png"
        ns["fixed_h0"].save_paper_figure(fig, out, dpi=200)
        print(summary[["panel", "group", "role", "n"]].to_string(index=False))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
