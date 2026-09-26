"""Integration checks against the original publication renderer and saved inputs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.colors import to_rgba
import plot_same_arch_mean_k15_vgg16_focus as focus


class VGG16FocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="vgg16_focus_test_")
        cls.original_plot = staticmethod(focus.base._plot_one)
        cls.original_save = staticmethod(focus.base.fixed_h0.save_square_figure)
        common = {"architectures": ["VGG16"], "save_pdf": False,
                  "save_png": False, "show_inline": True}
        with contextlib.redirect_stdout(io.StringIO()):
            original = focus.publication.default_config()
            original.update(common, output_dir=Path(cls.temp.name) / "original")
            _, cls.source_details, cls.source_figures = focus.publication.render(original)
            config = focus.default_config()
            cls.config = config
            config.update(common, output_dir=Path(cls.temp.name) / "filtered")
            _, cls.details, cls.figures = focus.render(config)

    @classmethod
    def tearDownClass(cls):
        for _, _, fig in cls.figures + cls.source_figures:
            focus.plt.close(fig)
        cls.temp.cleanup()

    def test_exact_visible_inventory(self):
        self.assertEqual(len(self.figures), 2)
        for dataset, arch, fig in self.figures:
            self.assertEqual(arch, "VGG16")
            self.assertIn(dataset, ("CF10", "CF100"))
            ax = fig.axes[0]
            artists = {item.get_gid(): item for item in ax.collections}
            self.assertEqual(set(artists), {
                "victim", "positive:RT-AL", "positive:DKD", "positive:Knockoff",
                "h0_references:VGG16", "h0_references:DeiT",
            })
            self.assertEqual(len(ax.collections), 6)
            self.assertEqual(len(ax.lines), 2)
            for name, item in artists.items():
                self.assertEqual(len(item.get_offsets()),
                                 12 if name.startswith("h0_references:") else 1)
            self.assertEqual(sum(len(item.get_offsets()) for item in ax.collections), 28)
            self.assertIsNone(ax.get_legend())

    def test_coordinates_and_boundaries_unchanged(self):
        for (_, _, fig), (_, _, original), detail, previous in zip(
                self.figures, self.source_figures, self.details, self.source_details):
            ax, old = fig.axes[0], original.axes[0]
            # Extra padding must expand, never crop, the original data range.
            for get in ("get_xlim", "get_ylim"):
                self.assertLess(getattr(ax, get)()[0], getattr(old, get)()[0])
                self.assertGreater(getattr(ax, get)()[1], getattr(old, get)()[1])
            np.testing.assert_array_equal(detail["victim"], previous["victim"])
            for method in ("RT-AL", "DKD", "Knockoff"):
                np.testing.assert_array_equal(detail["positive_means"][method],
                                              previous["positive_means"][method])
            # Source order is RN18, DeiT, VGG16; VGG16's negative follows its refs.
            by_gid = {item.get_gid(): item for item in ax.collections}
            for reference_arch, index in (("DeiT", 1), ("VGG16", 2)):
                reference = by_gid[f"h0_references:{reference_arch}"]
                source = old.collections[index]
                display = detail["reference_display"][reference_arch]
                indices = [item["source_index"] for item in display["shown"]]
                np.testing.assert_array_equal(reference.get_offsets(), source.get_offsets()[indices])
                self.assertEqual(display["full_fit_reference_count"], 15)
                self.assertEqual(len(display["shown"]), 12)
                self.assertEqual(len(display["hidden"]), 3)
                self.assertEqual(len({item["model_name"] for item in display["shown"] + display["hidden"]}), 15)
                np.testing.assert_array_equal(reference.get_offsets().min(axis=0), source.get_offsets().min(axis=0))
                np.testing.assert_array_equal(reference.get_offsets().max(axis=0), source.get_offsets().max(axis=0))
                boundary = next(line for line in ax.lines
                                if line.get_gid() == f"h0_boundary:{reference_arch}")
                np.testing.assert_array_equal(boundary.get_xydata(), old.lines[index].get_xydata())
                self.assertEqual(boundary.get_linestyle(), old.lines[index].get_linestyle())
                self.assertEqual(boundary.get_linewidth(), old.lines[index].get_linewidth())
                self.assertEqual(boundary.get_alpha(), old.lines[index].get_alpha())
            self.assertEqual(set(detail["negative_layers"]), {"VGG16", "DeiT"})
            self.assertFalse(any(layer["negative_mean_plotted"]
                                 for layer in detail["negative_layers"].values()))
            self.assertEqual(detail["mean_point_p_F"], {
                method: previous["mean_point_p_F"][method] for method in self.config["methods"]
            })

    def test_display_subset_is_repeatable_and_input_order_independent(self):
        for detail in self.details:
            for display in detail["reference_display"].values():
                full = sorted(display["shown"] + display["hidden"], key=lambda item: item["source_index"])
                xy = np.asarray([item["xy"] for item in full])
                names = [item["model_name"] for item in full]
                spans = [np.ptp(detail["axis_limits"][axis]) for axis in ("x", "y")]
                first = focus._reference_display_indices(xy, names, 12, spans)
                again = focus._reference_display_indices(xy, names, 12, spans)
                reversed_indices = focus._reference_display_indices(xy[::-1], names[::-1], 12, spans)
                np.testing.assert_array_equal(first, again)
                self.assertEqual({names[i] for i in first}, {names[::-1][i] for i in reversed_indices})
                self.assertEqual(first.tolist(), [item["source_index"] for item in display["shown"]])
                for count in (None, 15):
                    np.testing.assert_array_equal(focus._reference_display_indices(xy, names, count, spans),
                                                  np.arange(15))

    def test_display_subset_rejects_invalid_counts(self):
        xy = np.array([[0, 1], [1, 0], [2, 1], [1, 2], [1, 1]], dtype=float)
        names = [f"model_{index}" for index in range(len(xy))]
        for count in (0, 6, True, 2.5, 3):
            with self.assertRaises(ValueError):
                focus._reference_display_indices(xy, names, count, [2, 2])

    def test_new_reference_markers_colors_and_hidden_tick_labels(self):
        for (_, _, fig), detail in zip(self.figures, self.details):
            ax = fig.axes[0]
            artists = {item.get_gid(): item for item in ax.collections}
            self.assertEqual(ax.get_xticklabels(), [])
            self.assertEqual(ax.get_yticklabels(), [])
            self.assertEqual(ax.get_xlabel(), "I (X;T)")
            self.assertEqual(ax.get_ylabel(), "I (T;Y)")
            self.assertTrue(any(line.get_visible() for line in ax.get_xgridlines()))
            boundaries = {line.get_gid(): line for line in ax.lines}
            for arch, marker, area in (("VGG16", "o", 8), ("DeiT", "x", 16)):
                reference = artists[f"h0_references:{arch}"]
                expected = focus.styled._marker_path(marker)
                np.testing.assert_array_equal(reference.get_paths()[0].vertices, expected.vertices)
                np.testing.assert_array_equal(reference.get_paths()[0].codes, expected.codes)
                np.testing.assert_array_equal(reference.get_sizes(), [area])
                for color in (reference.get_facecolors(), reference.get_edgecolors()):
                    np.testing.assert_allclose(color, [to_rgba("black")])
                expected_color = {"VGG16": "#2878B5", "DeiT": "#8E63CE"}[arch]
                self.assertEqual(to_rgba(boundaries[f"h0_boundary:{arch}"].get_color()),
                                 to_rgba(expected_color))
                style = detail["architecture_styles"][arch]
                self.assertEqual(to_rgba(style["reference_color"]), to_rgba("black"))
                self.assertEqual(style["boundary_color"],
                                 self.config["architecture_styles"][arch]["boundary_color"])
            np.testing.assert_array_equal(artists["victim"].get_sizes(), [300])
            for method in ("RT-AL", "DKD", "Knockoff"):
                np.testing.assert_allclose(artists[f"positive:{method}"].get_sizes(),
                                           [80 * focus.base.METHOD_SIZE_SCALE[method]])

    def test_dejavu_sans_font(self):
        self.assertEqual(self.config["font_family"], "DejaVu Sans")
        for _, _, fig in self.figures:
            ax = fig.axes[0]
            labels = [ax.xaxis.label, ax.yaxis.label]
            labels += [tick.label1 for tick in ax.xaxis.get_major_ticks()]
            labels += [tick.label1 for tick in ax.yaxis.get_major_ticks()]
            for label in labels:
                self.assertEqual(label.get_fontfamily(), ["DejaVu Sans"])
                self.assertEqual(label.get_fontproperties().get_name(), "DejaVu Sans")

    def test_enlarged_markers_fit_inside_frame(self):
        for _, _, fig in self.figures:
            ax = fig.axes[0]
            for artist in ax.collections:
                path = artist.get_paths()[0]
                vertices = path.vertices * np.sqrt(artist.get_sizes()[0]) * fig.dpi / 72
                half_stroke = max(artist.get_linewidths(), default=0) * fig.dpi / 144
                low, high = vertices.min(axis=0) - half_stroke, vertices.max(axis=0) + half_stroke
                centers = artist.get_offset_transform().transform(artist.get_offsets())
                for center in centers:
                    self.assertTrue(np.all(center + low >= ax.bbox.p0))
                    self.assertTrue(np.all(center + high <= ax.bbox.p1))

    def test_hooks_restored_after_success_and_failure(self):
        self.assertIs(focus.base._plot_one, self.original_plot)
        self.assertIs(focus.base.fixed_h0.save_square_figure, self.original_save)
        with patch.object(focus.base, "render", side_effect=RuntimeError("export failed")):
            with self.assertRaisesRegex(RuntimeError, "export failed"):
                focus.render()
        self.assertIs(focus.base._plot_one, self.original_plot)
        self.assertIs(focus.base.fixed_h0.save_square_figure, self.original_save)


if __name__ == "__main__":
    unittest.main()
