"""Legend-only checks; no statistical render or source-data modification."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.colors import to_rgba
import pdfplumber

import plot_same_arch_mean_k15_vgg16_focus_shared_legend as legend


class FocusLegendTests(unittest.TestCase):
    def test_exact_labels_markers_and_colors(self):
        panel = legend.focus.default_config()
        handles, labels, entries = legend._legend_entries(panel, legend.default_config())
        self.assertEqual(labels, [
            "Victim ReNet-18", "Independent ResNet-18", "Independent DeiT",
            "Fine-tuning", "Knowledge distillation", "Model extraction",
        ])
        self.assertEqual([handle.get_marker() for handle in handles], ["*", "o", "x", "p", "v", ">"])
        expected_colors = [panel["colors"]["victim"], "black", "black"] + [
            panel["colors"]["positive_groups"][legend.focus.base.POSITIVE_GROUPS[method]]
            for method in panel["methods"]
        ]
        for handle, color in zip(handles, expected_colors):
            self.assertEqual(to_rgba(handle.get_color()), to_rgba(color))
            self.assertEqual(handle.get_linestyle(), "None")
        self.assertEqual([item["source_key"] for item in entries],
                         ["victim", "VGG16", "DeiT", "RT-AL", "DKD", "Knockoff"])

    def test_live_font_and_presentation_label_override(self):
        config = legend.default_config()
        self.assertEqual(config["font_family"], legend.focus.default_config()["font_family"])
        config["labels"]["references"]["VGG16"] = "Custom reference label"
        _, labels, entries = legend._legend_entries(legend.focus.default_config(), config)
        self.assertEqual(labels[1], "Custom reference label")
        self.assertEqual(entries[1]["source_key"], "VGG16")

    def test_export_without_panel_render(self):
        with tempfile.TemporaryDirectory(prefix="focus_legend_test_") as temp:
            config = legend.default_config()
            config["output_dir"] = Path(temp)
            figures_before = legend.plt.get_fignums()
            with patch.object(legend.focus, "render", side_effect=AssertionError("Must not render panels")):
                paths, manifest_path = legend.render(config)
            self.assertEqual(legend.plt.get_fignums(), figures_before)
            self.assertEqual({path.suffix for path in paths}, {".pdf", ".png"})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["entries"]), 6)
            self.assertTrue(manifest["display_labels_only"])
            with pdfplumber.open(next(path for path in paths if path.suffix == ".pdf")) as doc:
                self.assertEqual(len(doc.pages), 1)
                page = doc.pages[0]
                self.assertEqual(page.extract_text().splitlines(),
                                 [item["label"] for item in manifest["entries"]])
                self.assertTrue(all("DejaVuSans" in char["fontname"] for char in page.chars))
                self.assertLess(page.width, 180)
                self.assertLess(page.height, 120)


if __name__ == "__main__":
    unittest.main()
