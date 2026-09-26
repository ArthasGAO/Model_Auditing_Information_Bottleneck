"""Export right-side shared legends for the unknown-architecture sweep grids.

The sample-rate, bin-number, and k-number publication scripts all reuse the
same four curve styles.  Their per-panel legends are intentionally disabled;
this script exports compact, transparent legend-only PDFs for placement in a
right-hand LaTeX minipage.  The p-value legend includes the alpha threshold,
whereas the T^2 legend contains only the four positive forms.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D

import plot_unknown_arch_raw_sweeps_sample as base
import plot_unknown_arch_raw_sweeps_sample_publication_2x3 as publication


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = (
    ROOT
    / "saved_plots"
    / "unknown_arch_sweeps_publication_shared_legend_v1"
)

# Editable shared-legend settings.  These are deliberately larger than the
# former in-panel legend settings because LaTeX will place this narrow artwork
# beside an entire figure grid.
LEGEND_FONT_SIZE = 10.5
LEGEND_MARKER_SIZE = 7.5
LEGEND_LINE_WIDTH = 2.0
LEGEND_MARKER_EDGE_WIDTH = 0.45
LEGEND_HANDLE_LENGTH = 2.45
LEGEND_HANDLE_TEXT_PAD = 0.62
LEGEND_LABEL_SPACING = 0.58
FIGSIZE_INCHES = (1.72, 2.25)
PAD_INCHES = 0.02
PNG_DPI = 400

# One script serves both figure types.  Keep the threshold only in the p-value
# legend because T^2 panels do not draw an alpha line.
LEGEND_VARIANTS = {
    "pvalue": {"include_threshold": True},
    "t2": {"include_threshold": False},
}


def configure_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [base.FONT_FAMILY, "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def build_handles(include_threshold):
    handles = []
    labels = []

    for positive_form in base.POSITIVE_FORMS:
        style = base.CURVE_STYLES[positive_form]
        handles.append(Line2D(
            [], [],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=LEGEND_LINE_WIDTH,
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
        ))
        labels.append(positive_form)

    if include_threshold:
        handles.append(Line2D(
            [], [],
            color=base.THRESHOLD,
            linestyle="-",
            linewidth=LEGEND_LINE_WIDTH,
        ))
        labels.append(rf"$\alpha={base.ALPHA:g}$")

    return handles, labels


def render_variant(name, include_threshold):
    font = FontProperties(family=base.FONT_FAMILY, size=LEGEND_FONT_SIZE)
    findfont(FontProperties(family=base.FONT_FAMILY), fallback_to_default=False)

    fig = plt.figure(figsize=FIGSIZE_INCHES)
    fig.set_layout_engine(None)
    handles, labels = build_handles(include_threshold)
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=1,
        frameon=False,
        prop=font,
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
        borderaxespad=0.0,
        columnspacing=0.0,
    )

    stem = f"unknown_arch_sweeps_{name}_shared_legend_right"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=PAD_INCHES,
        transparent=True,
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=PAD_INCHES,
        transparent=True,
    )
    plt.close(fig)
    return pdf_path, png_path, labels


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for name, spec in LEGEND_VARIANTS.items():
        pdf_path, png_path, labels = render_variant(
            name,
            include_threshold=spec["include_threshold"],
        )
        outputs[name] = {
            "include_threshold": spec["include_threshold"],
            "labels": labels,
            "pdf": str(pdf_path.resolve()),
            "png": str(png_path.resolve()),
            "pdf_sha256": sha256(pdf_path),
        }

    source_paths = [
        Path(__file__),
        Path(base.__file__),
        Path(publication.__file__),
    ]
    manifest = {
        "description": (
            "Right-side shared legends for the sample-rate, bin-number, "
            "and k-number unknown-architecture publication grids"
        ),
        "render_script": str(Path(__file__).resolve()),
        "font_family": base.FONT_FAMILY,
        "font_size_points": LEGEND_FONT_SIZE,
        "marker_size_points": LEGEND_MARKER_SIZE,
        "line_width_points": LEGEND_LINE_WIDTH,
        "method_order": list(base.POSITIVE_FORMS),
        "curve_styles": base.CURVE_STYLES,
        "alpha": base.ALPHA,
        "threshold_color": base.THRESHOLD,
        "outputs": outputs,
        "source_hashes": {
            str(path.resolve()): sha256(path) for path in source_paths
        },
    }
    manifest_path = OUTPUT_DIR / "legend_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for variant in outputs.values():
        print(variant["pdf"])
        print(variant["png"])
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
