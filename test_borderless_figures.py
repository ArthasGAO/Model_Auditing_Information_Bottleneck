"""Regenerate all three figure variants and inspect every marker/legend stroke."""
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


class BorderlessTests(unittest.TestCase):
    def test_all_variants(self):
        ns, book = notebook_namespace()
        old = json.loads((ROOT/"tmp/distribution_check.before_borderless_markers.ipynb").read_text(encoding="utf-8"))
        config_ids = {"fixed-h0-config", "b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-single-config"}
        for before, after in zip(old["cells"], book["cells"]):
            if after.get("id") not in config_ids:
                self.assertEqual(before, after)
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config", "fixed-h0-grid-render"):
            execute_cell(old, identity, ns)
        baseline_points, baseline_summary = ns["grid_point_results"], ns["grid_summary"]
        baseline_axes = ns["axes_grid"]
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config",
                         "fixed-h0-single-config", "fixed-h0-overlay-config"):
            execute_cell(book, identity, ns)
        outputs = [ROOT/"saved_plots/extraction_grid_2x4.pdf"]
        outputs += [ROOT/ns["SINGLE_OUTPUT_DIR"]/(name+".pdf") for name in ns["SINGLE_FILENAMES"]]
        outputs += [ROOT/ns["OVERLAY_OUTPUT_DIR"]/(arch+"_all_methods.pdf") for arch in ns["OVERLAY_ARCH_NAMES"]]
        for path in outputs:
            relative = path.resolve().relative_to((ROOT/"saved_plots").resolve())
            backup = ROOT/"tmp/borderless_figure_backup"/relative
            if path.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(path.read_bytes())
        for identity in ("fixed-h0-grid-render", "fixed-h0-single-render", "fixed-h0-overlay-render"):
            source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == identity)
            with patch("matplotlib.pyplot.show"):
                exec(compile(source, identity, "exec"), ns)
        pd.testing.assert_frame_equal(ns["grid_point_results"], baseline_points)
        pd.testing.assert_frame_equal(ns["grid_summary"], baseline_summary)
        pd.testing.assert_frame_equal(ns["single_point_results"], baseline_points)
        for before, after in zip(baseline_axes.flat, ns["axes_grid"].flat):
            np.testing.assert_array_equal(before.lines[0].get_xydata(), after.lines[0].get_xydata())
            for c, p in zip(after.collections, before.collections):
                np.testing.assert_array_equal(c.get_offsets(), p.get_offsets())
                np.testing.assert_array_equal(c.get_sizes(), p.get_sizes())
        figures = [ns["fig_grid"], *ns["single_figures"], *ns["overlay_figures"]]
        for fig in figures:
            legends = list(fig.legends)
            for ax in fig.axes:
                for collection in ax.collections:
                    self.assertEqual(len(collection.get_edgecolors()), 0)
                    self.assertTrue(np.all(np.asarray(collection.get_linewidths()) == 0))
                for line in ax.lines:
                    self.assertGreater(line.get_linewidth(), 0)  # H0 boundary retained
                for spine in ax.spines.values():
                    self.assertGreater(spine.get_linewidth(), 0)
                if ax.get_legend() is not None:
                    legends.append(ax.get_legend())
            for legend in legends:
                for marker in legend.get_lines():
                    self.assertEqual(marker.get_markeredgewidth(), 0)
                    self.assertEqual(marker.get_markeredgecolor(), "none")
        self.assertTrue(all(p.is_file() for p in outputs))
        folder = ROOT/"tmp/fixed_h0_plot_validation/borderless"
        folder.mkdir(parents=True, exist_ok=True)
        for fig, name in ((figures[0], "grid"), (figures[1], "single"), (figures[-2], "overlay_RN18"), (figures[-1], "overlay_DeiT")):
            ns["fixed_h0"].save_paper_figure(fig, folder/(name+".png"), dpi=160, bbox_inches="tight", pad_inches=.02)
        print("PASS: 11 PDFs regenerated; scatter and legend edges all absent; data, H0 and axes borders unchanged.")
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
