"""Export three CF100/DeiT p-value sweeps on one identical PDF canvas.

This is a layout-validation exporter for the three panels currently placed in
one LaTeX row: dataset-usage rate, number of MI bins, and reference-pool size.
It reuses the live data loaders and plot functions from the three publication
scripts, but deliberately disables ``bbox_inches='tight'``.  Each PDF is
therefore exactly 2.25 x 2.25 inches, with the same axes rectangle expressed
in figure coordinates.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_unknown_arch_k_sweep_publication_2x3 as k_sweep
import plot_unknown_arch_raw_sweeps_bins_publication_2x3 as bin_sweep
import plot_unknown_arch_raw_sweeps_sample as sample_base
import plot_unknown_arch_raw_sweeps_sample_publication_2x3 as publication


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = (
    ROOT
    / "saved_plots"
    / "cf100_deit_three_sweeps_fixed_page_box_v2"
)

DATASET = "CF100"
ARCHITECTURE = "DeiT"
K_REF = 15
METRIC = "pvalue"
PNG_DPI = 400

# Fixed page and axes geometry shared by all three outputs.
PAGE_SIZE_INCHES = publication.PANEL_FIGSIZE_INCHES
# A single safe fixed-canvas rectangle is used for all three plots.  Its
# horizontal and vertical spans are both 0.695, so the axes remain square.
# Compared with the tight-export layout, the extra left margin prevents the
# rotated y label from touching the physical PDF page boundary.
AXES_MARGINS = {
    "left": 0.280,
    "right": 0.975,
    "bottom": 0.200,
    "top": 0.895,
}
SAVEFIG_KWARGS = {
    "bbox_inches": None,
    "pad_inches": 0.0,
    "facecolor": "white",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_panel(stem, plot_callback, title, subject):
    fig, ax = plt.subplots(figsize=PAGE_SIZE_INCHES)
    plot_callback(ax)
    fig.subplots_adjust(**AXES_MARGINS)

    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(
        pdf_path,
        metadata={"Title": title, "Subject": subject},
        **SAVEFIG_KWARGS,
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        **SAVEFIG_KWARGS,
    )
    plt.close(fig)
    return pdf_path, png_path


def main():
    publication.configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_rows, sample_sources = sample_base.load_all_rows()
    sample_scenarios = sample_base.build_scenarios(sample_rows)
    sample_key = (DATASET, ARCHITECTURE, K_REF)
    sample_base.require(sample_key in sample_scenarios,
                        f"Missing sample-rate scenario: {sample_key}")

    bin_scenarios, bin_sources = bin_sweep.load_bin_scenarios()
    sample_base.require(sample_key in bin_scenarios,
                        f"Missing bin-sweep scenario: {sample_key}")

    k_scenarios, k_sources = k_sweep.load_k_scenarios()
    k_key = (DATASET, ARCHITECTURE)
    sample_base.require(k_key in k_scenarios,
                        f"Missing k-sweep scenario: {k_key}")

    specifications = [
        {
            "sweep": "sample_rate",
            "stem": "cf100_deit_k15_pvalue_sample_rate_fixed_page_box",
            "plot": lambda ax: publication.plot_panel(
                ax, sample_scenarios[sample_key], METRIC, DATASET
            ),
            "sources": sample_sources,
        },
        {
            "sweep": "bins",
            "stem": "cf100_deit_k15_pvalue_bins_fixed_page_box",
            "plot": lambda ax: bin_sweep.plot_panel(
                ax, bin_scenarios[sample_key], METRIC, DATASET
            ),
            "sources": bin_sources,
        },
        {
            "sweep": "k_number",
            "stem": "cf100_deit_pvalue_k_number_fixed_page_box",
            "plot": lambda ax: k_sweep.plot_panel(
                ax, k_scenarios[k_key], METRIC, DATASET, ARCHITECTURE
            ),
            "sources": k_sources,
        },
    ]

    outputs = []
    for specification in specifications:
        pdf_path, png_path = export_panel(
            specification["stem"],
            specification["plot"],
            title=(
                f"{DATASET} {ARCHITECTURE} {METRIC} "
                f"{specification['sweep']} sweep"
            ),
            subject="Fixed-page-box publication panel",
        )
        outputs.append({
            "sweep": specification["sweep"],
            "pdf": str(pdf_path.resolve()),
            "png": str(png_path.resolve()),
            "pdf_sha256": sha256(pdf_path),
            "source_tables": specification["sources"],
        })

    source_scripts = [
        Path(__file__),
        Path(publication.__file__),
        Path(bin_sweep.__file__),
        Path(k_sweep.__file__),
    ]
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "CF100/DeiT p-value sample-rate, bin, and k-number panels "
            "exported on one identical fixed PDF page box"
        ),
        "selection": {
            "dataset": DATASET,
            "architecture": ARCHITECTURE,
            "k_ref_for_sample_and_bin": K_REF,
            "metric": METRIC,
        },
        "layout": {
            "page_size_inches": list(PAGE_SIZE_INCHES),
            "expected_page_size_points": [
                PAGE_SIZE_INCHES[0] * 72.0,
                PAGE_SIZE_INCHES[1] * 72.0,
            ],
            "axes_margins": AXES_MARGINS,
            "axes_box_aspect": 1.0,
            "bbox_inches": None,
            "pad_inches": 0.0,
            "show_legend": publication.SHOW_LEGEND,
        },
        "outputs": outputs,
        "source_hashes": {
            str(path.resolve()): sha256(path) for path in source_scripts
        },
    }
    manifest_path = OUTPUT_DIR / "fixed_page_box_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(OUTPUT_DIR.resolve())
    for output in outputs:
        print(output["pdf"])
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
