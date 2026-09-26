"""Filtered VGG16 variant of the existing Figure 1 publication plots.

Keep the victim, RT-AL/DKD/Knockoff means, and VGG16/DeiT H0 boundaries
and reference points. Remove every evaluated-negative mean and RN18 layer.
The original scripts, selected models and coordinates are kept unchanged.
The display hides numeric ticks and uses black reference points with distinct
shapes and colored H0 boundaries. Victim/positive markers are enlarged, with
extra edge padding for the star.
Reference thinning is display-only: H0 fitting still uses all 15 fixed models.
Edit default_config() here for subsequent changes to this variant.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_same_arch_mean_k15_publication_2x3 as publication


ROOT = Path(__file__).resolve().parent
styled = publication.styled
base = styled.base


def default_config():
    """Independent settings; inherit the established publication appearance."""
    config = publication.default_config()
    config["font_family"] = "DejaVu Sans"
    # Retain the full source scene when determining axes/ticks, before filtering.
    config["axis_extent_methods"] = list(config["methods"])
    config["datasets"] = ["CF10", "CF100"]
    config["architectures"] = ["VGG16"]
    config["methods"] = ["RT-AL", "DKD", "Knockoff"]
    config["reference_architectures"] = ["VGG16", "DeiT"]
    # Display-only counts. Set either value to None to show its full pool.
    config["reference_display_counts"] = {"VGG16": 12, "DeiT": 12}
    config["show_tick_labels"] = False
    config["reference_colors"] = {"VGG16": "#000000", "DeiT": "#000000"}
    config["architecture_styles"]["VGG16"].update({
        "reference_marker": "o", "reference_size_scale": 1.0,
        "reference_linewidth": 0.2, "boundary_color": "#2878B5",
    })
    config["architecture_styles"]["DeiT"].update({
        "reference_marker": "x", "reference_size_scale": 2.0,
        "reference_linewidth": 0.75, "boundary_color": "#8E63CE",
    })
    # Matplotlib scatter sizes represent area in points squared, not diameter.
    config["marker_area_multipliers"] = {"victim": 1.8, "positive": 1.6}
    for kind, multiplier in config["marker_area_multipliers"].items():
        config["sizes"][kind] *= multiplier
    # The larger victim star needs room at the upper-left data extreme.
    config["margins"]["data"] = 0.12
    config["output_dir"] = (
        ROOT / "saved_plots/same_arch_mean_k15_vgg16_focus_refs12_dejavu_sans_v5"
    )
    return config


def _reference_display_indices(xy, model_names, count, axis_spans):
    """Prune close non-extreme points deterministically, without moving any.

    Distance is measured in axes fractions. Protect x/y extrema, then remove
    the point with the closest remaining neighbor; ties favor removing a point
    nearer the full-pool centroid, then use immutable model names. Ordering by
    model identity first makes the selected identities invariant to input order.
    """
    xy = np.asarray(xy, dtype=float)
    spans = np.asarray(axis_spans, dtype=float)
    names = [str(name) for name in model_names]
    base.require(xy.shape == (len(names), 2) and len(names) > 0,
                 "Reference coordinates and model identities must match")
    base.require(np.isfinite(xy).all() and spans.shape == (2,)
                 and np.isfinite(spans).all() and np.all(spans > 0),
                 "Invalid coordinates or axis spans for reference display")
    base.require(len(set(names)) == len(names), "Reference model identities must be unique")
    if count is None:
        return np.arange(len(xy))
    base.require(isinstance(count, (int, np.integer)) and not isinstance(count, (bool, np.bool_))
                 and 1 <= count <= len(xy), "Display count must be an integer within the full pool")
    order = np.array(sorted(range(len(names)), key=names.__getitem__))
    points = (xy[order] - xy[order[0]]) / spans
    protected = set(np.argmin(points, axis=0)) | set(np.argmax(points, axis=0))
    base.require(count >= len(protected), "Display count is too small to preserve x/y extrema")
    distances = np.sum((points[:, None] - points[None, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    centrality = np.sum((points - points.mean(axis=0)) ** 2, axis=1)
    remaining = list(range(len(points)))
    while len(remaining) > count:
        candidates = [index for index in remaining if index not in protected]
        drop = min(candidates, key=lambda index: (
            float(distances[index, remaining].min()),
            float(centrality[index]), names[order[index]],
        ))
        remaining.remove(drop)
    return np.sort(order[remaining])


def _thin_reference_display(reference, dataset, reference_arch, detail, config, limits):
    # Reload via the authoritative loader to obtain the immutable model names
    # and seeds in exactly the same order as the already validated scatter.
    layer = detail["negative_layers"][reference_arch]
    scenario = layer["h0_scenario"]
    null = base.fixed_h0.load_nulls(
        [scenario], bins=config["bins"], in_size=config["in_size"], k=config["k"],
        round_id=layer["round_id"], alpha=config["alpha"],
        result_dir=Path(config["result_root"]) / dataset / "negatives",
    )[scenario]
    rows = null.reference
    xy = rows[["I(X;T)-In", "I(T;Y)-In"]].to_numpy(dtype=float)
    np.testing.assert_array_equal(np.asarray(reference.get_offsets()), xy)
    names = rows.model_name.astype(str).tolist()
    selected = _reference_display_indices(
        xy, names, config["reference_display_counts"][reference_arch],
        [np.ptp(limits["x"]), np.ptp(limits["y"])],
    )
    reference.set_offsets(xy[selected])
    selected_set = set(selected.tolist())

    def record(index):
        return {"source_index": int(index), "model_name": names[index],
                "seed": int(rows.iloc[index].seed), "xy": xy[index].tolist()}

    layer["displayed_reference_count"] = len(selected)
    return {
        "selection": "prune_closest_non_extreme_points",
        "display_only": True,
        "full_fit_reference_count": len(xy),
        "displayed_reference_count": len(selected),
        "shown": [record(index) for index in selected],
        "hidden": [record(index) for index in range(len(xy)) if index not in selected_set],
    }


def _focused_plot_one(dataset, arch, config, table_rows,
                      positive_models, victims, selections):
    source_config = deepcopy(config)
    source_config["methods"] = list(config["axis_extent_methods"])
    fig, detail = styled._styled_plot_one(
        dataset, arch, source_config, table_rows, positive_models, victims, selections)
    ax = fig.axes[0]
    original_limits = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())}
    collections = list(ax.collections)
    boundaries = list(ax.lines)
    keep_arches = set(config["reference_architectures"])
    keep_methods = set(config["methods"])
    detail["reference_display"] = {}

    # The inherited renderer validates the layer order and number of artists.
    index = 0
    for boundary, reference_arch in zip(boundaries, styled._overlay_arches(arch, source_config)):
        reference = collections[index]
        index += 1
        if reference_arch in keep_arches:
            boundary.set_gid(f"h0_boundary:{reference_arch}")
            reference.set_gid(f"h0_references:{reference_arch}")
            color = config["reference_colors"][reference_arch]
            reference.set_facecolor(color)
            reference.set_edgecolor(color)
            detail["architecture_styles"][reference_arch]["reference_color"] = color
            detail["reference_display"][reference_arch] = _thin_reference_display(
                reference, dataset, reference_arch, detail, config, original_limits)
        else:
            boundary.remove()
            reference.remove()
        if source_config["show_nonmatching_negative_means"] or reference_arch == arch:
            collections[index].remove()
            index += 1

    for method in source_config["methods"]:
        positive = collections[index]
        index += 1
        if method in keep_methods:
            positive.set_gid(f"positive:{method}")
        else:
            positive.remove()
    base.require(index == len(collections) - 1, "Unexpected victim artist position")
    collections[index].set_gid("victim")
    # Keep axis titles, grid positions and tick marks; hide only the numbers.
    ax.tick_params(axis="both", which="both",
                   labelbottom=config["show_tick_labels"],
                   labelleft=config["show_tick_labels"])
    ax.xaxis.offsetText.set_visible(config["show_tick_labels"])
    ax.yaxis.offsetText.set_visible(config["show_tick_labels"])

    # Update provenance to describe the visible scene, not the hidden source scene.
    detail["h0_boundaries"] = [item for item in detail["h0_boundaries"]
                               if item["architecture"] in keep_arches]
    for key in ("negative_layers", "architecture_styles"):
        detail[key] = {name: values for name, values in detail[key].items()
                       if name in keep_arches}
        for values in detail[key].values():
            values["negative_mean_plotted"] = False
            if "negative_mean_size" in values:
                values["negative_mean_size"] = None
    for key in ("positive_means", "mean_point_p_F"):
        detail[key] = {method: detail[key][method] for method in config["methods"]}
    detail.pop("negative_mean", None)
    detail["axis_limits"] = original_limits
    detail["show_tick_labels"] = bool(config["show_tick_labels"])
    detail["visible_elements"] = {
        "victim_points": 1,
        "positive_methods": list(config["methods"]),
        "positive_points": len(keep_methods),
        "reference_architectures": list(config["reference_architectures"]),
        "reference_points": sum(len(artist.get_offsets()) for artist in ax.collections
                                if artist.get_gid().startswith("h0_references:")),
        "h0_boundaries": len(ax.lines),
        "negative_mean_points": 0,
    }

    expected_gids = ({"victim"} | {f"positive:{name}" for name in keep_methods}
                     | {f"h0_references:{name}" for name in keep_arches})
    base.require({artist.get_gid() for artist in ax.collections} == expected_gids,
                 "Unexpected visible scatter groups")
    base.require(len(ax.collections) == len(expected_gids), "Unexpected scatter count")
    base.require({line.get_gid() for line in ax.lines}
                 == {f"h0_boundary:{name}" for name in keep_arches},
                 "Unexpected visible H0 boundaries")
    base.require(np.array_equal(ax.get_xlim(), original_limits["x"])
                 and np.array_equal(ax.get_ylim(), original_limits["y"]),
                 "Filtering changed the original axis ranges")
    fig.canvas.draw()
    return fig, detail


def render(config=None):
    config = deepcopy(default_config() if config is None else config)
    base.require(config["architectures"] == ["VGG16"], "This variant is for VGG16 cases only")
    base.require(config["show_all_arch_negative_overlays"] and not config["show_legend"],
                 "Keep the original full source scene without a panel legend")
    base.require(set(config["methods"]) <= set(config["axis_extent_methods"]),
                 "Visible methods must belong to the original source scene")
    base.require(set(config["reference_architectures"]) <= set(base.ALL_ARCHITECTURES),
                 "Unknown reference architecture")
    base.require(set(config["reference_architectures"]) <= set(config["reference_colors"]),
                 "Set a color for each visible reference architecture")
    base.require(set(config["reference_architectures"]) <= set(config["reference_display_counts"]),
                 "Set a display count for each visible reference architecture")
    base.require(isinstance(config["show_tick_labels"], (bool, np.bool_)),
                 "show_tick_labels must be boolean")
    padding = float(config["tight_pad_inches"])
    base.require(np.isfinite(padding) and padding >= 0, "Invalid tight padding")

    def save_figure(fig, path, dpi=300):
        if config["save_tight"]:
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=padding)
        else:
            styled._BASE_SAVE_SQUARE_FIGURE(fig, path, dpi=dpi)

    # Scope the in-memory hooks to this render, including on export failure.
    # No original source files or original output paths are modified.
    original_plot = base._plot_one
    original_save = base.fixed_h0.save_square_figure
    base._plot_one = _focused_plot_one
    base.fixed_h0.save_square_figure = save_figure
    try:
        output_paths, details, figures = base.render(config)
    finally:
        base._plot_one = original_plot
        base.fixed_h0.save_square_figure = original_save

    manifest_path = Path(config["output_dir"]) / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "description": "VGG16 cases: victim, selected positive means, VGG16/DeiT references and H0 boundaries; no negative squares",
        "render_script": str(Path(__file__).resolve()),
        "negative_overlay_mode": "selected_reference_architectures",
        "negative_mean_mode": "hidden",
        "reference_architectures": list(config["reference_architectures"]),
        "visible_positive_methods": list(config["methods"]),
        "axis_extent_methods": list(config["axis_extent_methods"]),
        "axis_extent_policy": "full original scene data, with configured edge padding",
        "data_padding_fraction": config["margins"]["data"],
        "reference_colors": deepcopy(config["reference_colors"]),
        "boundary_colors": {arch: config["architecture_styles"][arch]["boundary_color"]
                            for arch in config["reference_architectures"]},
        "reference_display_counts": deepcopy(config["reference_display_counts"]),
        "reference_display_note": (
            "Only a subset of reference points is displayed for readability. "
            "H0 fits, boundaries and p-values retain all 15 original fixed reference models; "
            "per-figure reference_display records shown and hidden model identities."
        ),
        "marker_area_multipliers": deepcopy(config["marker_area_multipliers"]),
        "marker_sizes_points_squared": deepcopy(config["sizes"]),
        "boundary_display_scales": {arch: config["boundary_display_scales"][arch]
                                    for arch in config["reference_architectures"]},
        "publication_layout": {
            "panel_figsize_inches": list(config["figsize"]),
            "font_family": config["font_family"],
            "font_sizes_points": deepcopy(config["font_sizes"]),
            "tight_crop": bool(config["save_tight"]),
            "tight_pad_inches": padding,
            "png_dpi": config["png_dpi"],
            "show_tick_labels": bool(config["show_tick_labels"]),
        },
        "outputs": [str(path.resolve()) for path in output_paths],
    })
    for source in (__file__, publication.__file__, styled.__file__, base.__file__,
                   base.fixed_h0.__file__, base.audit.__file__):
        path = Path(source).resolve()
        manifest["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return output_paths, details, figures


if __name__ == "__main__":
    paths, details, _ = render()
    for detail in details:
        counts = ", ".join(
            f"{arch}={item['displayed_reference_count']}/{item['full_fit_reference_count']}"
            for arch, item in detail["reference_display"].items()
        )
        print(f"{detail['dataset']}: displayed/full-fit reference points: {counts}")
    for path in paths:
        print(path)
