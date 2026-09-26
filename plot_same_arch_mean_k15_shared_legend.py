"""Generate one shared legend for the publication-size 2x3 MI figure grid.

The legend reads the live publication configuration so architecture colors,
reference markers, positive-method markers and victim styling stay synchronized
with the six exported panels. It writes a narrow transparent PDF for LaTeX and
a PNG preview; it does not modify or regenerate the six data panels.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D

import plot_same_arch_mean_k15 as base
import plot_same_arch_mean_k15_arch_styled_negatives as styled
import plot_same_arch_mean_k15_publication_2x3 as publication


ROOT = Path(__file__).resolve().parent


def default_config():
    return {
        "output_dir": ROOT / "saved_plots/same_arch_mean_k15_publication_2x3_three_h0_single_negative_scaled_boundaries_shared_legend",
        "stem": "same_arch_mean_k15_shared_legend",
        "figsize": (1.45, 2.80),
        "font_family": "Times New Roman",
        "font_size": 8.0,
        "architecture_marker_size": 4.8,
        "negative_mean_marker_size": 5.2,
        "positive_marker_size": 5.2,
        "victim_marker_size": 6.5,
        "labelspacing": 0.32,
        "pad_inches": 0.015,
        "png_dpi": 400,
        "save_pdf": True,
        "save_png": True,
    }


def _architecture_handles(panel_config, legend_config):
    reference_handles, reference_labels = [], []
    negative_handles, negative_labels = [], []
    styles = panel_config["architecture_styles"]
    for arch in base.ALL_ARCHITECTURES:
        style = styles[arch]
        display_arch = "VG16" if arch == "VGG16" else arch
        reference_handles.append(Line2D(
            [], [],
            color=panel_config["colors"]["reference"],
            linestyle="None",
            marker=style["reference_marker"],
            markerfacecolor=panel_config["reference_facecolor"],
            markeredgecolor=panel_config["colors"]["reference"],
            markeredgewidth=style["reference_linewidth"],
            markersize=legend_config["architecture_marker_size"],
        ))
        reference_labels.append(f"{display_arch} references")

        if (panel_config["show_all_arch_negative_overlays"]
                and panel_config["show_nonmatching_negative_means"]):
            negative_handles.append(Line2D(
                [], [],
                color=style.get("boundary_color", style["negative_color"]),
                linestyle=panel_config["boundary_linestyle"],
                linewidth=panel_config["boundary_linewidth"],
                marker="s",
                markerfacecolor=style["negative_color"],
                markeredgecolor="none",
                markersize=legend_config["negative_mean_marker_size"],
            ))
            negative_labels.append(f"{display_arch} negatives")

    if not panel_config["show_nonmatching_negative_means"]:
        style = styles[base.ALL_ARCHITECTURES[0]]
        negative_handles.append(Line2D(
            [], [],
            color=style.get("boundary_color", panel_config["colors"]["boundary"]),
            linestyle=panel_config["boundary_linestyle"],
            linewidth=panel_config["boundary_linewidth"],
            marker="s",
            markerfacecolor=style["negative_color"],
            markeredgecolor="none",
            markersize=legend_config["negative_mean_marker_size"],
        ))
        negative_labels.append("Negatives")

    return (
        reference_handles + negative_handles,
        reference_labels + negative_labels,
    )


def _suspect_handles(panel_config, legend_config):
    handles = [Line2D(
        [], [],
        color=panel_config["colors"]["victim"],
        linestyle="None",
        marker=panel_config["markers"]["victim"],
        markerfacecolor=panel_config["colors"]["victim"],
        markeredgecolor="none",
        markersize=legend_config["victim_marker_size"],
    )]
    labels = ["Victim"]
    for method in panel_config["methods"]:
        group = base.POSITIVE_GROUPS[method]
        handles.append(Line2D(
            [], [],
            color=panel_config["colors"]["positive_groups"][group],
            linestyle="None",
            marker=panel_config["markers"]["positive"][method],
            markerfacecolor=panel_config["colors"]["positive_groups"][group],
            markeredgecolor="none",
            markersize=legend_config["positive_marker_size"],
        ))
        labels.append(method)
    return handles, labels


def render(config=None):
    config = default_config() if config is None else dict(config)
    panel_config = publication.default_config()
    font = FontProperties(family=config["font_family"], size=config["font_size"])
    findfont(FontProperties(family=config["font_family"]), fallback_to_default=False)

    fig = plt.figure(figsize=config["figsize"])
    fig.set_layout_engine(None)

    arch_handles, arch_labels = _architecture_handles(panel_config, config)
    suspect_handles, suspect_labels = _suspect_handles(panel_config, config)
    fig.legend(
        arch_handles + suspect_handles,
        arch_labels + suspect_labels,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=False,
        prop=font,
        handlelength=2.0,
        handletextpad=0.55,
        borderaxespad=0,
        labelspacing=config["labelspacing"],
    )

    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = config["stem"]
    output_paths = []
    if config["save_pdf"]:
        path = out / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=config["pad_inches"],
                    transparent=True)
        output_paths.append(path)
    if config["save_png"]:
        path = out / f"{stem}.png"
        fig.savefig(path, dpi=config["png_dpi"], bbox_inches="tight",
                    pad_inches=config["pad_inches"], transparent=True)
        output_paths.append(path)

    sources = [
        Path(__file__),
        Path(base.__file__),
        Path(styled.__file__),
        Path(publication.__file__),
    ]
    manifest = {
        "description": "Single shared right-side legend synchronized with the publication panel configuration",
        "render_script": str(Path(__file__).resolve()),
        "architecture_order": list(base.ALL_ARCHITECTURES),
        "method_order": list(panel_config["methods"]),
        "architecture_styles": panel_config["architecture_styles"],
        "boundary_display_scales": panel_config["boundary_display_scales"],
        "positive_group_colors": panel_config["colors"]["positive_groups"],
        "source_hashes": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "outputs": [str(path.resolve()) for path in output_paths],
    }
    manifest_path = out / "legend_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    return output_paths, manifest_path


if __name__ == "__main__":
    render()
