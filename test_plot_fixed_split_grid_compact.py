"""Real-data layout/export regression checks, with no MI or H0 changes."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell

ROOT = Path(__file__).resolve().parent


class CompactGridTests(unittest.TestCase):
    def assert_caption_clearance(self, fig, axes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for i in range(8):
            caption = next(t for t in fig.texts if t.get_gid() == f"panel-caption-{i+1}")
            bounds = caption.get_window_extent(renderer)
            ax = axes.flat[i]
            # Caption must be below all tick/axis labels of its own panel.
            decorations = ax.get_tightbbox(renderer)
            self.assertLess(bounds.y1, decorations.y0)
            if i < 4:
                below = axes.flat[i+4].get_tightbbox(renderer)
                self.assertGreater(bounds.y0, below.y1)

    def test_compact_export_and_unchanged_data(self):
        ns, book = notebook_namespace()
        old = json.loads((ROOT / "tmp/distribution_check.before_compact_grid.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(book["cells"]), len(old["cells"]))
        for before, after in zip(old["cells"], book["cells"]):
            if after.get("id") not in ("fixed-h0-grid-config", "fixed-h0-grid-render"):
                self.assertEqual(before, after)
        execute_cell(book, "b6bf5527-38a7-43f3-b5c0-ab17d70cb169", ns)
        execute_cell(old, "fixed-h0-grid-config", ns)
        execute_cell(old, "fixed-h0-grid-render", ns)
        old_fig, old_axes = ns["fig_grid"], ns["axes_grid"]
        old_points, old_summary = ns["grid_point_results"], ns["grid_summary"]
        execute_cell(book, "fixed-h0-grid-config", ns)
        execute_cell(book, "fixed-h0-grid-render", ns)
        fig, axes = ns["fig_grid"], ns["axes_grid"]
        pd.testing.assert_frame_equal(ns["grid_point_results"], old_points)
        pd.testing.assert_frame_equal(ns["grid_summary"], old_summary)
        self.assertAlmostEqual(old_fig.get_figheight()-fig.get_figheight(), .21)
        for ax, prev in zip(axes.flat, old_axes.flat):
            np.testing.assert_array_equal(ax.get_xlim(), prev.get_xlim())
            np.testing.assert_array_equal(ax.get_ylim(), prev.get_ylim())
            self.assertAlmostEqual(ax.bbox.width, ax.bbox.height)
            self.assertAlmostEqual(ax.bbox.width, prev.bbox.width)
            for c, p in zip(ax.collections, prev.collections):
                np.testing.assert_array_equal(c.get_offsets(), p.get_offsets())
                np.testing.assert_array_equal(c.get_sizes(), p.get_sizes())
                np.testing.assert_array_equal(c.get_facecolors(), p.get_facecolors())
            for c, p in zip(ax.lines, prev.lines):
                np.testing.assert_array_equal(c.get_xydata(), p.get_xydata())
        self.assert_caption_clearance(fig, axes)
        self.assertFalse(fig.artists, "No invisible full-canvas rectangle")
        self.assertEqual([t.get_text() for t in fig.texts], [t.get_text() for t in old_fig.texts])
        renderer = fig.canvas.get_renderer()
        crop = fig.get_tightbbox(renderer)
        self.assertLess(crop.height+.04, fig.get_figheight())
        # Public helper forwards explicit tight/padding settings and keeps default compatibility.
        with patch.object(fig, "savefig") as save:
            ns["fixed_h0"].save_paper_figure(fig, "unused.pdf", bbox_inches="tight", pad_inches=.02)
            self.assertEqual(save.call_args.kwargs["bbox_inches"], "tight")
            self.assertEqual(save.call_args.kwargs["pad_inches"], .02)
        output = ROOT / "saved_plots/extraction_grid_2x4.pdf"
        if output.exists():
            backup = ROOT / "tmp/extraction_grid_2x4.before_compact.pdf"
            if not backup.exists():
                backup.write_bytes(output.read_bytes())
        ns["fixed_h0"].save_paper_figure(fig, output, bbox_inches="tight", pad_inches=.02)
        preview = ROOT / "tmp/fixed_h0_plot_validation/grid_compact.png"
        ns["fixed_h0"].save_paper_figure(fig, preview, dpi=200, bbox_inches="tight", pad_inches=.02)
        old_gap = (old_axes[0, 0].bbox.y0-old_axes[1, 0].bbox.y1)/old_fig.dpi
        new_gap = (axes[0, 0].bbox.y0-axes[1, 0].bbox.y1)/fig.dpi
        print(f"Axes row gap: {old_gap:.4f} -> {new_gap:.4f} inches")
        print(f"Canvas: {fig.get_figwidth():.4f} x {fig.get_figheight():.4f} inches")
        print(f"Tight export: {crop.width+.04:.4f} x {crop.height+.04:.4f} inches")
        print(f"Saved {output}")
        # Future multi-line captions also remain clear at the smallest row reserve.
        ns["GRID_CAPTIONS"] = [text+"\nSynthetic preview" for text in ns["GRID_CAPTIONS"]]
        ns["GRID_ROW_GAP_IN"] = 0
        execute_cell(book, "fixed-h0-grid-render", ns)
        self.assert_caption_clearance(ns["fig_grid"], ns["axes_grid"])
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
