"""Independent-panel export matches the grid's points and exact H0 curves."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text
from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell

ROOT = Path(__file__).resolve().parent


class IndividualPanelTests(unittest.TestCase):
    def test_export_and_configuration(self):
        ns, book = notebook_namespace()
        old = json.loads((ROOT/"tmp/distribution_check.before_individual_panels.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(book["cells"][:-3], old["cells"])
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config",
                         "fixed-h0-grid-render", "fixed-h0-single-config"):
            execute_cell(book, identity, ns)
        source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == "fixed-h0-single-render")
        with patch("matplotlib.pyplot.show"):
            exec(compile(source, "independent-panel-export", "exec"), ns)
        pd.testing.assert_frame_equal(ns["single_point_results"], ns["grid_point_results"])
        pd.testing.assert_frame_equal(ns["single_summary"], ns["grid_summary"])
        self.assertEqual(len(ns["single_pdf_paths"]), 8)
        for i, (fig, ax, path) in enumerate(zip(ns["single_figures"], ns["single_axes"], ns["single_pdf_paths"])):
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(ax.get_title(), "")
            self.assertFalse(fig.texts)
            self.assertFalse(fig.artists)
            self.assertAlmostEqual(ax.bbox.width, ax.bbox.height)
            self.assertEqual(ax.get_xlabel(), "I (X;T)")
            self.assertEqual(ax.get_ylabel(), "I (T;Y)")
            self.assertEqual(ax.get_legend()._loc, 0)
            labels = [t.get_text() for t in ax.get_legend().texts]
            expected = ["Victim model", "positive suspects", "negative suspects", "reference models"]
            if i in (2, 6):
                expected.remove("positive suspects")
            self.assertEqual(labels, expected)
            legend_box = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
            self.assertGreaterEqual(legend_box.x0, ax.bbox.x0)
            self.assertLessEqual(legend_box.x1, ax.bbox.x1)
            self.assertGreaterEqual(legend_box.y0, ax.bbox.y0)
            self.assertLessEqual(legend_box.y1, ax.bbox.y1)
            old_ax = ns["axes_grid"].flat[i]
            self.assertEqual(len(ax.collections), len(old_ax.collections))
            for new, previous in zip(ax.collections, old_ax.collections):
                np.testing.assert_array_equal(new.get_offsets(), previous.get_offsets())
            for new, previous in zip(ax.lines, old_ax.lines):
                np.testing.assert_array_equal(new.get_xydata(), previous.get_xydata())
            for text in fig.findobj(Text):
                if text.get_text():
                    self.assertEqual(text.get_fontproperties().get_name(), "Times New Roman")
        previews = ROOT/"tmp/fixed_h0_plot_validation/individual"
        previews.mkdir(parents=True, exist_ok=True)
        ns["fixed_h0"].save_panel_figures(ns["single_figures"],
            [previews/(name+".png") for name in ns["SINGLE_FILENAMES"]], dpi=150)
        # Prove external font and marker controls reach the rendered objects.
        fig, ax, _, _ = ns["fixed_h0"].plot_fixed_split_single(
            ns["PLOT_PANELS"][0], framework=ns,
            font_sizes=dict(xlabel=12, ylabel=11, xtick=9, ytick=10, legend=6),
            role_styles={"positive_style": {"s": 31}},
            legend_options={"loc": "lower left"})
        self.assertEqual(ax.xaxis.label.get_fontsize(), 12)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 11)
        self.assertEqual(ax.get_legend().texts[0].get_fontsize(), 6)
        self.assertEqual(ax.get_legend()._loc, 3)
        self.assertEqual(ax.collections[2].get_sizes()[0], 31)
        print("Eight PDFs saved. Exact grid point/curve agreement; best legends inside; no captions.")
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
