"""Publication-size k=15 MI panels for a two-row by three-column LaTeX grid.

This variant leaves the existing plotting scripts untouched. It reuses the
current architecture colors, reference markers, marker scaling, linewidths,
data validation and tight export, but renders each panel close to its intended
physical size in the paper so LaTeX does not heavily shrink every visual mark.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import plot_same_arch_mean_k15_arch_styled_negatives as styled


ROOT = Path(__file__).resolve().parent


def default_config():
    """Return settings for one panel in a full-width 2x3 paper figure."""
    config = styled.default_config()
    config["output_dir"] = (
        ROOT / "saved_plots/same_arch_mean_k15_publication_2x3_three_h0_single_negative_scaled_boundaries_tight"
    )
    # Keep all three architecture-specific H0 pools in every panel, but show
    # only the evaluated-negative mean matched to the panel architecture.
    config["show_all_arch_negative_overlays"] = True
    config["show_nonmatching_negative_means"] = False
    config["boundary_display_scales"] = {
        "RN18": 1.6,
        "VGG16": 1.6,
        "DeiT": 1.0,
    }
    for style in config["architecture_styles"].values():
        style["negative_color"] = config["colors"]["negative"]
        style["boundary_color"] = config["colors"]["reference"]
    config["figsize"] = (2.25, 2.25)
    config["font_sizes"] = {
        "xlabel": 15,
        "ylabel": 15,
        "ticks": 12,
        "legend": 6.5,
    }
    config["sizes"] = {
        "victim": 120,
        "negative": 35,
        "reference": 8,
        "positive": 50,
    }
    # The x6 reference scaling compensates for shrinking a 5.2-inch source
    # panel. A near-final-size 2.25-inch panel needs substantially less.
    config["architecture_styles"]["VGG16"]["reference_size_scale"] = 3.0
    config["architecture_styles"]["VGG16"]["reference_linewidth"] = 0.7
    config["architecture_styles"]["DeiT"]["reference_size_scale"] = 3.0
    config["architecture_styles"]["DeiT"]["reference_linewidth"] = 0.7
    config["png_dpi"] = 400
    config["save_tight"] = True
    config["tight_pad_inches"] = 0.015
    return config


def render(config=None):
    """Render six publication-size panels and record their intended layout."""
    config = deepcopy(default_config() if config is None else config)
    output_paths, details, figures = styled.render(config)

    manifest_path = Path(config["output_dir"]) / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["style_render_script"] = manifest.get("render_script")
    manifest["render_script"] = str(Path(__file__).resolve())
    manifest["source_hashes"][str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    manifest["publication_layout"] = {
        "grid": "2x3",
        "panel_figsize_inches": list(config["figsize"]),
        "latex_panel_width": "0.322\\textwidth",
        "font_sizes_points": deepcopy(config["font_sizes"]),
        "tight_crop": bool(config["save_tight"]),
        "tight_pad_inches": float(config["tight_pad_inches"]),
        "png_dpi": int(config["png_dpi"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_paths, details, figures


if __name__ == "__main__":
    render()
