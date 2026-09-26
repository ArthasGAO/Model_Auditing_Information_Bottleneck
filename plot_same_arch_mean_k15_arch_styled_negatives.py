"""Architecture-styled negative overlays for the k=15 same-architecture plots.

This is an additive rendering variant. It reuses the validated data selection,
F-test boundaries, source checks and export path from
``plot_same_arch_mean_k15.py`` without changing that script.

Architecture identity is encoded twice within each negative layer:

* the H0 boundary and evaluated-negative mean square share one color;
* the H0 reference models use one filled marker shape at the original size.

The victim and positive means remain specific to the figure architecture and
retain the original styling.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt

import plot_same_arch_mean_k15 as base


ROOT = Path(__file__).resolve().parent

ARCHITECTURE_STYLES = {
    "RN18": {
        "negative_color": "#1686D9",
        "reference_marker": "o",
        "reference_size_scale": 1.0,
        "reference_linewidth": 0.2,
    },
    "VGG16": {
        "negative_color": "#7A5AA6",
        "reference_marker": "2",
        "reference_size_scale": 4.0,
        "reference_linewidth": 0.6,
    },
    "DeiT": {
        "negative_color": "#2A9D8F",
        "reference_marker": "x",
        "reference_size_scale": 4.0,
        "reference_linewidth": 0.6,
    },
}

_BASE_PLOT_ONE = base._plot_one
_BASE_SAVE_SQUARE_FIGURE = base.fixed_h0.save_square_figure


def default_config():
    """Return an editable config while leaving the base script untouched."""
    config = base.default_config()
    config["output_dir"] = (
        ROOT / "saved_plots/same_arch_mean_k15_arch_styled_negatives_scaled_bold_refs_tight"
    )
    config["architecture_styles"] = deepcopy(ARCHITECTURE_STYLES)
    config["show_all_arch_negative_overlays"] = True
    config["show_legend"] = False
    config["save_tight"] = True
    config["tight_pad_inches"] = 0.02
    return config


def _overlay_arches(arch, config):
    arches = (list(base.ALL_ARCHITECTURES)
              if config["show_all_arch_negative_overlays"] else [arch])
    return [name for name in arches if name != arch] + [arch]


def _marker_path(marker):
    style = MarkerStyle(marker)
    return style.get_path().transformed(style.get_transform())


def _styled_plot_one(dataset, arch, config, table_rows,
                     positive_models, victims, selections):
    """Render with the base implementation, then style only negative layers."""
    fig, details = _BASE_PLOT_ONE(
        dataset, arch, config, table_rows, positive_models, victims, selections)
    ax = fig.axes[0]
    arches = _overlay_arches(arch, config)
    styles = config["architecture_styles"]

    base.require(set(styles) == set(base.ALL_ARCHITECTURES),
                 "architecture_styles must define RN18, VGG16 and DeiT exactly")
    base.require(len(ax.lines) == len(arches),
                 f"{dataset} {arch}: unexpected boundary artist count")
    negative_mean_count = sum(
        config["show_nonmatching_negative_means"] or negative_arch == arch
        for negative_arch in arches
    )
    expected_collections = len(arches) + negative_mean_count + len(config["methods"]) + 1
    base.require(len(ax.collections) == expected_collections,
                 f"{dataset} {arch}: unexpected scatter artist count")

    applied = {}
    collection_index = 0
    for index, negative_arch in enumerate(arches):
        style = styles[negative_arch]
        color = style["negative_color"]
        boundary_color = style.get("boundary_color", color)
        marker = style["reference_marker"]
        reference_size_scale = float(style.get("reference_size_scale", 1.0))
        reference_linewidth = float(
            style.get("reference_linewidth", config["reference_linewidth"]))
        base.require(isinstance(color, str) and color,
                     f"{negative_arch}: negative_color must be a nonempty string")
        base.require(isinstance(boundary_color, str) and boundary_color,
                     f"{negative_arch}: boundary_color must be a nonempty string")
        base.require(math.isfinite(reference_size_scale) and reference_size_scale > 0,
                     f"{negative_arch}: reference_size_scale must be finite and positive")
        base.require(math.isfinite(reference_linewidth) and reference_linewidth > 0,
                     f"{negative_arch}: reference_linewidth must be finite and positive")

        boundary = ax.lines[index]
        reference = ax.collections[collection_index]
        collection_index += 1
        negative_mean = None
        if config["show_nonmatching_negative_means"] or negative_arch == arch:
            negative_mean = ax.collections[collection_index]
            collection_index += 1

        boundary.set_color(boundary_color)
        reference.set_paths([_marker_path(marker)])
        reference_base_size = float(reference.get_sizes()[0])
        reference.set_sizes([reference_base_size * reference_size_scale])
        reference.set_linewidths([reference_linewidth])
        reference.set_facecolor(config["reference_facecolor"])
        reference.set_edgecolor(config["colors"]["reference"])
        if negative_mean is not None:
            negative_mean.set_facecolor(color)
            negative_mean.set_edgecolor("none")

        # set_paths/set_facecolor do not change the configured scatter sizes.
        applied[negative_arch] = {
            "negative_color": color,
            "boundary_color": boundary_color,
            "reference_marker": marker,
            "reference_base_size": reference_base_size,
            "reference_size_scale": reference_size_scale,
            "reference_size": float(reference.get_sizes()[0]),
            "reference_linewidth": float(reference.get_linewidths()[0]),
            "negative_mean_marker": "s",
            "negative_mean_plotted": negative_mean is not None,
            "negative_mean_size": (
                float(negative_mean.get_sizes()[0]) if negative_mean is not None else None
            ),
        }

    details["architecture_styles"] = applied
    fig.canvas.draw()
    return fig, details


def render(config=None):
    """Render the isolated style variant and record the mapping in its manifest."""
    config = deepcopy(default_config() if config is None else config)
    base.require(not config["show_legend"],
                 "This first architecture-style preview keeps the existing no-legend layout")
    base.require(isinstance(config["save_tight"], (bool,)),
                 "save_tight must be boolean")
    tight_pad_inches = float(config["tight_pad_inches"])
    base.require(math.isfinite(tight_pad_inches) and tight_pad_inches >= 0,
                 "tight_pad_inches must be finite and nonnegative")

    def save_figure(fig, path, dpi=300):
        if not config["save_tight"]:
            return _BASE_SAVE_SQUARE_FIGURE(fig, path, dpi=dpi)
        with plt.rc_context({"savefig.bbox": "tight"}):
            fig.savefig(path, dpi=dpi, bbox_inches="tight",
                        pad_inches=tight_pad_inches)

    original_plot_one = base._plot_one
    original_save_figure = base.fixed_h0.save_square_figure
    base._plot_one = _styled_plot_one
    base.fixed_h0.save_square_figure = save_figure
    try:
        output_paths, details, figures = base.render(config)
    finally:
        base._plot_one = original_plot_one
        base.fixed_h0.save_square_figure = original_save_figure

    manifest_path = Path(config["output_dir"]) / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["render_script"] = str(Path(__file__).resolve())
    manifest["architecture_styles"] = deepcopy(config["architecture_styles"])
    manifest["export_crop"] = {
        "bbox_inches": "tight" if config["save_tight"] else None,
        "pad_inches": tight_pad_inches if config["save_tight"] else None,
    }
    manifest["source_hashes"][str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_paths, details, figures


if __name__ == "__main__":
    render()
