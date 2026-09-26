"""Geometry regression tests using the existing, validated sweep inputs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
from PIL import Image

import plot_unknown_arch_sweeps_shared_tight as shared


class SharedTightExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = shared.prepare_shared_export()

    def test_union_contains_all_panels_and_does_not_change_curves(self):
        for family, module in self.plan.modules.items():
            scenarios, _ = self.plan.data[family]
            for metric in module.METRICS_TO_PLOT:
                common = self.plan.boxes[metric]
                for key, curves in scenarios.items():
                    fig = module.create_panel(key[0], key[1], curves, metric)
                    try:
                        ax = fig.axes[0]
                        before = [line.get_xydata().copy() for line in ax.lines]
                        original_size = fig.get_size_inches().copy()
                        original_dpi = fig.dpi
                        content, _ = shared.measure_content(fig, module.PNG_DPI)
                        self.assertLessEqual(common.x0, content.x0 - shared.SHARED_PAD_INCHES + 1e-9)
                        self.assertLessEqual(common.y0, content.y0 - shared.SHARED_PAD_INCHES + 1e-9)
                        self.assertGreaterEqual(common.x1, content.x1 + shared.SHARED_PAD_INCHES - 1e-9)
                        self.assertGreaterEqual(common.y1, content.y1 + shared.SHARED_PAD_INCHES - 1e-9)
                        np.testing.assert_array_equal(fig.get_size_inches(), original_size)
                        self.assertEqual(fig.dpi, original_dpi)
                        for previous, line in zip(before, ax.lines):
                            np.testing.assert_array_equal(previous, line.get_xydata())
                    finally:
                        plt.close(fig)

    def test_standalone_save_uses_common_pdf_and_png_geometry(self):
        with tempfile.TemporaryDirectory(prefix="sweep_shared_tight_") as tempdir:
            png_sizes = {}
            for family, module in self.plan.modules.items():
                scenarios, _ = self.plan.data[family]
                for metric in module.METRICS_TO_PLOT:
                    key = (("CF100", "DeiT") if family == "reference_pool"
                           else ("CF100", "DeiT", 15))
                    args = (*key, scenarios[key], metric)
                    # Exercise the individual save API without an explicit plan.
                    with patch.object(module, "OUTPUT_DIR", Path(tempdir) / family), \
                         patch.object(module, "prepare_shared_export", return_value=self.plan) as prepare:
                        pdf, png = module.save_panel(*args)
                        prepare.assert_called_once()
                    with pdfplumber.open(pdf) as document:
                        self.assertEqual(len(document.pages), 1)
                        page = document.pages[0]
                        np.testing.assert_allclose(
                            [page.width, page.height],
                            self.plan.metadata["page_size_points_by_metric"][metric], atol=1e-6,
                        )
                        spines = [line for line in page.lines
                                  if line["height"] > 100 and line["width"] < 1e-6
                                  and abs(line["linewidth"] - 0.72) < 1e-6]
                        self.assertEqual(len(spines), 2)
                        actual = [min(line["x0"] for line in spines), spines[0]["y0"],
                                  max(line["x0"] for line in spines), spines[0]["y1"]]
                        np.testing.assert_allclose(
                            actual, self.plan.metadata["axes_extents_points_by_metric"][metric],
                            atol=1e-6,
                        )
                    with Image.open(png) as image:
                        expected = png_sizes.setdefault(metric, image.size)
                        self.assertEqual(image.size, expected)
                        image.verify()

    def test_incompatible_axes_geometry_is_rejected(self):
        module = self.plan.modules["bins"]
        original = module.create_panel

        def changed_panel(*args):
            fig = original(*args)
            fig.subplots_adjust(left=0.3)
            return fig

        with patch.object(module, "create_panel", side_effect=changed_panel):
            with self.assertRaisesRegex(ValueError, "Axes geometry differs"):
                shared.prepare_shared_export()


if __name__ == "__main__":
    unittest.main()
