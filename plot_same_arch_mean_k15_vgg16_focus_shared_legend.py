"""Standalone six-entry legend for the filtered VGG16 information plane.

Read live marker/color/font settings from the focus script, without loading
experiment data or regenerating any panels. Labels are presentation-only:
the user-requested Independent ResNet-18 label refers to the VGG16 marker.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D

import plot_same_arch_mean_k15_shared_legend as shared
import plot_same_arch_mean_k15_vgg16_focus as focus


ROOT = Path(__file__).resolve().parent


def default_config():
    config = shared.default_config()
    config.update({
        "output_dir": ROOT / "saved_plots/same_arch_mean_k15_vgg16_focus_shared_legend_v1",
        "stem": "vgg16_focus_shared_legend",
        "figsize": (2.4, 1.3),
        "font_family": focus.default_config()["font_family"],
        "labelspacing": 0.4,
        "labels": {
            "victim": "Victim ReNet-18",
            "references": {
                "VGG16": "Independent ResNet-18",
                "DeiT": "Independent DeiT",
            },
            "positives": {
                "RT-AL": "Fine-tuning",
                "DKD": "Knowledge Distillation",
                "Knockoff": "Model Extraction",
            },
        },
    })
    return config


def _legend_entries(panel_config, config):
    """Return handles, labels and explicit source-key/display-label mappings."""
    suspect_handles, methods = shared._suspect_handles(panel_config, config)
    handles = [suspect_handles[0]]
    labels = [config["labels"]["victim"]]
    sources = [("victim", "victim")]
    for arch in panel_config["reference_architectures"]:
        style = panel_config["architecture_styles"][arch]
        color = panel_config["reference_colors"][arch]
        handles.append(Line2D(
            [], [], linestyle="None", marker=style["reference_marker"],
            color=color, markerfacecolor=color, markeredgecolor=color,
            markeredgewidth=style["reference_linewidth"],
            markersize=config["architecture_marker_size"],
        ))
        labels.append(config["labels"]["references"][arch])
        sources.append(("reference", arch))
    for handle, method in zip(suspect_handles[1:], methods[1:]):
        handles.append(handle)
        labels.append(config["labels"]["positives"][method])
        sources.append(("positive", method))
    entries = [
        {"kind": kind, "source_key": key, "label": label,
         "marker": handle.get_marker(), "color": handle.get_color(),
         "marker_size_points": handle.get_markersize()}
        for handle, label, (kind, key) in zip(handles, labels, sources)
    ]
    return handles, labels, entries


def render(config=None):
    config = deepcopy(default_config() if config is None else config)
    panel_config = focus.default_config()
    font = FontProperties(family=config["font_family"], size=config["font_size"])
    font_path = findfont(font, fallback_to_default=False)
    handles, labels, entries = _legend_entries(panel_config, config)
    fig = plt.figure(figsize=config["figsize"])
    fig.set_layout_engine(None)
    output_paths = []
    try:
        fig.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(0, 1),
            frameon=False, prop=font, ncol=1, handlelength=2.0,
            handletextpad=0.55, borderaxespad=0,
            labelspacing=config["labelspacing"],
        )
        out = Path(config["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            if config[f"save_{extension}"]:
                path = out / f"{config['stem']}.{extension}"
                fig.savefig(path, dpi=config["png_dpi"], bbox_inches="tight",
                            pad_inches=config["pad_inches"], transparent=True)
                output_paths.append(path)
        sources = [Path(__file__), Path(focus.__file__), Path(shared.__file__),
                   Path(focus.publication.__file__), Path(focus.styled.__file__),
                   Path(focus.base.__file__)]
        manifest = {
            "description": "Standalone continuous six-entry legend; no data panels modified",
            "render_script": str(Path(__file__).resolve()),
            "font_family": config["font_family"],
            "font_file": font_path,
            "font_size_points": config["font_size"],
            "entries": entries,
            "display_labels_only": True,
            "label_note": (
                "Labels follow the requested spelling verbatim. Independent ResNet-18 "
                "is the display label for the VGG16 reference marker; no underlying "
                "model identities or experiment data are changed."
            ),
            "tight_crop": True,
            "transparent": True,
            "pad_inches": config["pad_inches"],
            "source_hashes": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            "outputs": [str(path.resolve()) for path in output_paths],
        }
        manifest_path = out / "legend_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    finally:
        plt.close(fig)
    return output_paths, manifest_path


if __name__ == "__main__":
    paths, manifest = render()
    for path in [*paths, manifest]:
        print(path)
