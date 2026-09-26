"""Check method overlays reuse anchors once and preserve all source MI points."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell

ROOT = Path(__file__).resolve().parent


class OverlayTests(unittest.TestCase):
    def test_overlay(self):
        ns, book = notebook_namespace()
        old = json.loads((ROOT/"tmp/distribution_check.before_method_overlay.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(book["cells"][:-3], old["cells"])
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config",
                         "fixed-h0-grid-render", "fixed-h0-single-config", "fixed-h0-overlay-config"):
            execute_cell(book, identity, ns)
        panels_before = deepcopy(ns["PLOT_PANELS"])
        source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == "fixed-h0-overlay-render")
        with patch("matplotlib.pyplot.show"):
            exec(compile(source, "overlay-render", "exec"), ns)
        self.assertEqual(panels_before, ns["PLOT_PANELS"])
        self.assertEqual(len(ns["overlay_figures"]), 2)
        self.assertEqual(len(ns["overlay_point_results"]), 400)
        for row, arch in enumerate(("RN18", "DeiT")):
            fig, ax = ns["overlay_figures"][row], ns["overlay_axes"][row]
            original = ns["axes_grid"].flat[4*row]
            self.assertEqual([len(c.get_offsets()) for c in ax.collections], [50, 30, 50, 50, 50, 1])
            self.assertEqual(len(ax.lines), 1)
            np.testing.assert_array_equal(ax.lines[0].get_xydata(), original.lines[0].get_xydata())
            for idx in (0, 1, -1):
                np.testing.assert_array_equal(ax.collections[idx].get_offsets(), original.collections[idx].get_offsets())
            for j, col in enumerate((0, 1, 3)):
                grid = ns["axes_grid"].flat[4*row+col]
                np.testing.assert_array_equal(ax.collections[j+2].get_offsets(), grid.collections[2].get_offsets())
                np.testing.assert_allclose(ax.collections[j+2].get_facecolors()[0],
                                           to_rgba(ns["OVERLAY_METHOD_COLORS"][col], .85))
            expected = ns["grid_point_results"]
            expected = expected[(expected.panel.between(4*row+1, 4*row+4)) &
                                ((expected.role == "positive") | (expected.panel == 4*row+1))]
            actual = ns["overlay_point_results"].query("architecture == @arch")
            columns = [c for c in expected.columns if c != "panel"]
            pd.testing.assert_frame_equal(actual[columns].sort_values(["role", "group", "model_name"]).reset_index(drop=True),
                                          expected[columns].sort_values(["role", "group", "model_name"]).reset_index(drop=True))
            self.assertEqual([t.get_text() for t in ax.get_legend().texts],
                             ["Victim model", "FT-AL", "PR80%", "Knockoff", "negative suspects", "reference models"])
            self.assertEqual(ax.get_legend()._loc, 0)
            self.assertFalse(fig.texts)
            self.assertAlmostEqual(ax.bbox.width, ax.bbox.height)
            self.assertTrue(ns["overlay_pdf_paths"][row].is_file())
        # Original single-panel API still has its role-only legend.
        fig, ax, _, _ = ns["fixed_h0"].plot_fixed_split_single(ns["PLOT_PANELS"][0], framework=ns)
        self.assertEqual([t.get_text() for t in ax.get_legend().texts],
                         ["Victim model", "positive suspects", "negative suspects", "reference models"])
        bad = deepcopy(ns["PLOT_PANELS"])
        bad[1]["negative_groups"] = bad[4]["negative_groups"]
        with self.assertRaisesRegex(ValueError, "different negative_groups"):
            ns["fixed_h0"].combine_grid_row(bad, 0)
        preview = ROOT/"tmp/fixed_h0_plot_validation/overlay"
        preview.mkdir(parents=True, exist_ok=True)
        ns["fixed_h0"].save_panel_figures(ns["overlay_figures"],
            [preview/f"{arch}.png" for arch in ("RN18", "DeiT")], dpi=160)
        print("PASS: two overlays, 150 positives + 50 negatives per architecture; references and victim unchanged.")
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
