"""Real-data checks for fixed-H0 notebook plotting (no source data writes)."""
import json
from pathlib import Path
import sys
import types
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_fixed_split_h0 as plotting

BASE = Path(__file__).resolve().parent


def notebook_namespace():
    book = json.loads((BASE / "distribution_check.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in reversed(book["cells"])
                  if "class TableGroupSpec:" in "".join(c["source"]))
    module = types.ModuleType("h0_notebook_validation")
    sys.modules[module.__name__] = module
    exec(compile(source, "notebook-framework", "exec"), module.__dict__)
    module.display = lambda value: None
    config = next("".join(c["source"]) for c in book["cells"] if c.get("id") == "fixed-h0-config")
    exec(compile("\n".join(line for line in config.splitlines() if not line.startswith("%")),
                 "notebook-h0-config", "exec"), module.__dict__)
    return module.__dict__, book


class H0PlotTests(unittest.TestCase):
    def assert_square_presentation(self, fig, labels):
        from matplotlib.text import Text
        fig.canvas.draw()
        self.assertAlmostEqual(*fig.get_size_inches())
        self.assertFalse(fig.texts)
        for ax in fig.axes:
            self.assertEqual(ax.get_title(), "")
            box = ax.get_window_extent()
            self.assertAlmostEqual(box.width, box.height)
            legend = ax.get_legend()
            self.assertEqual(legend._loc, 0)  # Matplotlib's best location.
            self.assertEqual([t.get_text() for t in legend.get_texts()], labels)
        for text in fig.findobj(Text):
            if text.get_text():
                self.assertEqual(text.get_fontproperties().get_name(), "Times New Roman")

    @classmethod
    def setUpClass(cls):
        cls.ns, cls.book = notebook_namespace()
        cls.nulls = plotting.load_nulls([g["scenario"] for g in cls.ns["H0_GROUP_CATALOG"].values()])

    def tearDown(self):
        plt.close("all")

    def test_original_cells_unchanged(self):
        old = json.loads((BASE/"tmp/distribution_check.before_fixed_h0_20260921.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(self.book["cells"][:200], old["cells"][:200])
        self.assertEqual(self.book["metadata"], old["metadata"])

    def test_all_six_saved_groups_and_f_boundary(self):
        for null in self.nulls.values():
            self.assertEqual((len(null.reference), len(null.evaluation)), (30, 50))
            self.assertFalse(set(null.reference.seed) & set(null.evaluation.seed))
            self.assertAlmostEqual(null.cutoff_md2, 11.671881648048359)
            curve = null.boundary()
            np.testing.assert_allclose(null.p_values(curve), .01, rtol=1e-10, atol=1e-12)
            self.assertTrue((null.p_values(null.mu+(curve-null.mu)*1.001) < .01).all())
            self.assertTrue((null.p_values(null.mu+(curve-null.mu)*.999) > .01).all())
            self.assertEqual(sum(null.p_values(null.evaluation[["I(X;T)-In", "I(T;Y)-In"]]) < .01), 0)

    def test_execute_actual_notebook_cells(self):
        render = next("".join(c["source"]) for c in self.book["cells"] if c.get("id") == "fixed-h0-render")
        exec(compile(render, "notebook-h0-render", "exec"), self.ns)
        summary = self.ns["h0_summary"]
        self.assertEqual(len(summary), 6)
        self.assertEqual(summary.n.tolist(), [50]*6)
        self.assertEqual(len(self.ns["h0_point_results"]), 300)
        self.assertEqual(summary.query("role == 'negative'").outside.tolist(), [0]*3)
        self.assertEqual(len(self.ns["ax_h0"].lines), 3)
        from matplotlib.markers import MarkerStyle
        def count_points(marker, rgb):
            m = MarkerStyle(marker)
            expected = m.get_path().transformed(m.get_transform()).vertices
            count = 0
            for collection in self.ns["ax_h0"].collections:
                vertices = collection.get_paths()[0].vertices
                colors = collection.get_facecolors()
                if (vertices.shape == expected.shape and np.allclose(vertices, expected)
                        and len(colors) and np.allclose(colors[0, :3], rgb)):
                    count += len(collection.get_offsets())
            return count
        self.assertTrue(self.ns["SHOW_REFERENCE"])
        self.assertEqual(count_points("s", (0, 0, 0)), 1)
        self.assertEqual(count_points("^", (1, 0, 0)), 150)
        self.assertEqual(count_points("^", (0, 128/255, 0)), 150)
        self.assertEqual(count_points("o", (31/255, 119/255, 180/255)), 90)
        for zoom_ax in self.ns["fig_h0_zoom"].axes:
            self.assertEqual([len(c.get_offsets()) for c in zoom_ax.collections], [50, 30, 50])
            labels = [t.get_text() for t in zoom_ax.get_legend().get_texts()]
            self.assertEqual(labels, ["negative suspects", "reference models"])
        self.assert_square_presentation(self.ns["fig_h0"],
                                       ["Victim model", "positive suspects", "negative suspects", "reference models"])
        self.assert_square_presentation(self.ns["fig_h0_zoom"], ["negative suspects", "reference models"])
        out = BASE/"tmp/fixed_h0_plot_validation"
        out.mkdir(exist_ok=True)
        self.ns["fig_h0"].savefig(out/"main.png", dpi=140, bbox_inches="tight")
        self.ns["fig_h0_zoom"].savefig(out/"zoom.png", dpi=140, bbox_inches="tight")
        print(summary.to_string(index=False))

    def test_custom_positive_and_negative_seed_subsets(self):
        ns = self.ns
        negative = dict(ns["H0_GROUP_CATALOG"]["c10_rn18"])
        null = self.nulls[negative["scenario"]]
        negative["seeds"] = null.evaluation.seed.tolist()[:3]
        positive = ns["positive_group"]("subset", "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18",
                                         "c10_rn18", "purple", seeds=[0, 1, 2])
        _, _, summary, detail, _ = plotting.plot_fixed_split_plane(
            {"c10_rn18": negative}, [positive], [], framework=ns, zoom=False, show=False)
        self.assertEqual(summary.n.tolist(), [3, 3])
        self.assertEqual(summary.iloc[0].full_evaluation_n, 50)
        self.assertEqual(len(detail), 6)

    def test_reject_reference_as_evaluation_and_wrong_h0(self):
        negative = dict(self.ns["H0_GROUP_CATALOG"]["c10_rn18"])
        negative["seeds"] = [int(self.nulls[negative["scenario"]].reference.seed.iloc[0])]
        with self.assertRaisesRegex(ValueError, "never reference"):
            plotting.plot_fixed_split_plane({"c10_rn18": negative}, [], [], framework=self.ns, show=False)
        with self.assertRaisesRegex(ValueError, "wrong suspect"):
            plotting.check_knockoff_mapping(
                [{"scenario": "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Cross16"}],
                "CIFAR-10_ResNet-18_25000")

    def test_missing_positive_is_an_error(self):
        spec = self.ns["positive_group"]("missing", "does-not-exist", "c10_rn18", "red")["spec"]
        with self.assertRaisesRegex(ValueError, "no matching"):
            plotting.selected_points(spec, self.ns, 50, 25000)

    def test_user_single_rn18_copy_uses_requested_role_styles(self):
        from matplotlib.markers import MarkerStyle
        ns, book = notebook_namespace()
        for name in ("VICTIM_STYLE", "SUSPECT_STYLE", "POSITIVE_STYLE", "NEGATIVE_STYLE", "REFERENCE_STYLE"):
            ns.pop(name, None)  # The active copy must define its own settings.
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "d8db5a5f-be4d-4966-8bae-95027a8f4577"):
            source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == identity)
            source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
            exec(compile(source, identity, "exec"), ns)
        self.assertEqual(ns["ENABLED_NEGATIVE_GROUPS"], ["c10_rn18"])
        self.assertEqual(len(ns["POSITIVE_GROUPS"]), 1)
        self.assertTrue(ns["SHOW_REFERENCE"])
        self.assertEqual(ns["h0_summary"].n.tolist(), [50, 50])
        self.assertEqual(len(ns["h0_point_results"]), 100)
        for marker, color, number in [("s", (0, 0, 0), 1), ("^", (1, 0, 0), 50),
                                       ("^", (0, 128/255, 0), 50),
                                       ("o", (31/255, 119/255, 180/255), 30)]:
            m = MarkerStyle(marker)
            expected = m.get_path().transformed(m.get_transform()).vertices
            count = 0
            for collection in ns["ax_h0"].collections:
                vertices = collection.get_paths()[0].vertices
                if (vertices.shape == expected.shape and np.allclose(vertices, expected)
                        and np.allclose(collection.get_facecolors()[0, :3], color)):
                    count += len(collection.get_offsets())
            self.assertEqual(count, number)
        self.assert_square_presentation(ns["fig_h0"],
                                       ["Victim model", "positive suspects", "negative suspects", "reference models"])
        self.assert_square_presentation(ns["fig_h0_zoom"], ["negative suspects", "reference models"])
        zoom = ns["fig_h0_zoom"].axes[0]
        np.testing.assert_allclose(zoom.collections[0].get_facecolors()[0, :3], [0, 128/255, 0])
        np.testing.assert_allclose(zoom.collections[2].get_facecolors()[0, :3], [1, 0, 0])
        out = BASE/"tmp/fixed_h0_plot_validation"
        out.mkdir(exist_ok=True)
        ns["fig_h0"].savefig(out/"active_rn18_main.png", dpi=140, bbox_inches="tight")
        ns["fig_h0_zoom"].savefig(out/"active_rn18_zoom.png", dpi=140, bbox_inches="tight")
        from PIL import Image
        for name in ("active_rn18_main.png", "active_rn18_zoom.png"):
            with Image.open(out/name) as image:
                self.assertEqual(image.width, image.height)
        with plt.rc_context({"savefig.bbox": "tight"}):
            plotting.save_square_figure(ns["fig_h0"], out/"exact_square.png", dpi=100)
        with Image.open(out/"exact_square.png") as image:
            self.assertEqual(image.size, (600, 600))

    def test_cifar100_all_nine_positive_groups(self):
        import pandas as pd
        ns = self.ns
        names = pd.read_csv(ns["CSV_EX"])["Scenario"].unique()
        selected = [name for name in names if name.startswith("CIFAR-100_") and "_Knockoff_" in name]
        negative = {key: value for key, value in ns["H0_GROUP_CATALOG"].items() if key.startswith("c100_")}
        positives = []
        for name in selected:
            suffix = name.rsplit("_", 1)[-1]
            key = "c100_" + ({"Same18": "rn18", "Cross18": "rn18",
                              "Same16": "vgg16", "Cross16": "vgg16",
                              "SameDeiT": "deit", "CrossDeiT": "deit"}[suffix])
            positives.append(ns["positive_group"](name, name, key, "purple"))
        self.assertEqual(len(positives), 9)
        _, _, summary, details, _ = plotting.plot_fixed_split_plane(
            negative, positives, [], framework=ns, zoom=False, show=False)
        self.assertEqual(summary.n.tolist(), [50]*12)
        self.assertEqual(len(details), 600)
        self.assertEqual(summary.query("role == 'negative'").outside.sum(), 0)

    def test_paper_grid_notebook_cells(self):
        from matplotlib.text import Text
        from PIL import Image
        from unittest.mock import patch
        ns, book = notebook_namespace()
        for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169",
                         "fixed-h0-grid-config", "fixed-h0-grid-render"):
            source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == identity)
            source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
            # Do not overwrite a user-enabled export while validating the cell.
            with patch("matplotlib.figure.Figure.savefig"):
                exec(compile(source, identity, "exec"), ns)
        fig, axes = ns["fig_grid"], ns["axes_grid"]
        self.assertEqual(axes.shape, (2, 4))
        self.assertEqual(len(fig.axes), 8)
        self.assertEqual(len(plt.get_fignums()), 1)
        expected_height = 3.81 + 2*(8*1.25/72 + .08)
        np.testing.assert_allclose(fig.get_size_inches(), [6.5, expected_height])
        captions = [t for t in fig.texts if (t.get_gid() or "").startswith("panel-caption-")]
        self.assertEqual([t.get_text() for t in captions], ns["GRID_CAPTIONS"])
        for caption, ax in zip(captions, axes.flat):
            bounds = caption.get_window_extent()
            self.assertGreaterEqual(bounds.y0, 0)
            self.assertLess(bounds.y1, ax.get_window_extent().y0)
            self.assertAlmostEqual((bounds.x0+bounds.x1)/2,
                                   (ax.get_window_extent().x0+ax.get_window_extent().x1)/2)
            self.assertFalse(bounds.overlaps(ax.xaxis.label.get_window_extent()))
        self.assertEqual(len(ns["grid_point_results"]), 800)
        self.assertEqual(ns["grid_summary"].n.tolist(), [50]*16)
        self.assertEqual(ns["grid_summary"].query("role == 'negative'").outside.sum(), 0)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                         ["Victim model", "positive suspects", "negative suspects", "reference models"])
        for ax in axes.flat:
            self.assertEqual(ax.get_title(), "")
            self.assertIsNone(ax.get_legend())
            self.assertEqual(len(ax.child_axes), 1)
            inset = ax.child_axes[0]
            self.assertTrue(inset.get_gid().startswith("negative-inset-"))
            for main_line, inset_line in zip(ax.lines, inset.lines):
                np.testing.assert_array_equal(main_line.get_xydata(), inset_line.get_xydata())
                self.assertEqual(main_line.get_linewidth(), ns["H0_BOUNDARY_WIDTH"])
            for main_points, inset_points in zip(ax.collections, inset.collections):
                np.testing.assert_array_equal(main_points.get_offsets(), inset_points.get_offsets())
                np.testing.assert_allclose(inset_points.get_sizes(),
                                           main_points.get_sizes()*ns["INSET_MARKER_SCALE"]**2)
            box = ax.get_window_extent()
            self.assertAlmostEqual(box.width, box.height)
            self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), 131)
            np.testing.assert_allclose(ax.collections[0].get_offsets(), axes[0, 0].collections[0].get_offsets())
        for text in fig.findobj(Text):
            if text.get_text():
                self.assertEqual(text.get_fontproperties().get_name(), "Times New Roman")
        self.assertIsNot(ns["GRID_PANELS"][0]["negative_groups"], ns["GRID_PANELS"][1]["negative_groups"])
        old = json.loads((BASE/"tmp/distribution_check.before_marker_scale_20260921.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(book["cells"][:-3], old["cells"][:-3])
        out = BASE/"saved_plots"
        out.mkdir(exist_ok=True)
        for extension in ("png", "pdf"):
            plotting.save_paper_figure(fig, out/f"extraction_grid_2x4_template.{extension}", dpi=300)
        with Image.open(out/"extraction_grid_2x4_template.png") as image:
            self.assertEqual(image.width, 1950)
            self.assertLessEqual(abs(image.height-expected_height*300), 1)
        # More caption lines increase figure height, never shrink the panels.
        long_captions = ["(a) Knockoff\nResNet-18"] + ["A longer editable extraction caption"]*7
        fig_long, axes_long, _, _ = plotting.plot_fixed_split_grid(
            ns["GRID_PANELS"], framework=ns, panel_captions=long_captions, show=False)
        self.assertGreater(fig_long.get_size_inches()[1], expected_height)
        self.assertAlmostEqual(axes_long[0, 0].get_window_extent().width,
                               axes[0, 0].get_window_extent().width)
        for caption in fig_long.texts:
            bounds = caption.get_window_extent()
            self.assertGreaterEqual(bounds.y0, 0)
            self.assertGreaterEqual(bounds.x0, 0)
            self.assertLessEqual(bounds.x1, fig_long.bbox.width)
        self.assertEqual(fig_long.texts[0].get_text(), long_captions[0])
        # Default API scale is 1; notebook scale affects areas, never data.
        for scaled_ax, base_ax in zip(axes.flat, axes_long.flat):
            for scaled, base in zip(scaled_ax.collections, base_ax.collections):
                np.testing.assert_allclose(scaled.get_sizes(), base.get_sizes()*ns["MARKER_SCALE"]**2)
                np.testing.assert_allclose(scaled.get_offsets(), base.get_offsets())
            np.testing.assert_allclose(scaled_ax.lines[0].get_xydata(), base_ax.lines[0].get_xydata())
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            plotting.plot_fixed_split_grid(ns["GRID_PANELS"], framework=ns, marker_scale=0, show=False)
        with self.assertRaisesRegex(ValueError, "eight strings"):
            plotting.plot_fixed_split_grid(ns["GRID_PANELS"], framework=ns,
                                           panel_captions=["only one"], show=False)


if __name__ == "__main__":
    unittest.main()
