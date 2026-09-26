"""Shared tight cropping for all three unknown-architecture sweep families.

Run this script to export the complete collection into a new timestamped folder.
The three individual publication scripts also use this preparation step, so
running them separately still produces matching page boxes for each metric.
Only presentation geometry is shared; data loading and plotting remain in the
existing generic scripts. A common crop requires all three families' input data.
"""

import importlib
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "saved_plots" / "unknown_arch_three_sweeps_shared_tight_v1"
# Shared padding outside the union of all visible content, in inches.
SHARED_PAD_INCHES = 0.02
MODULE_NAMES = {
    "sample_rate": "plot_unknown_arch_raw_sweeps_sample_publication_2x3",
    "bins": "plot_unknown_arch_raw_sweeps_bins_publication_2x3",
    "reference_pool": "plot_unknown_arch_k_sweep_publication_2x3",
}


@dataclass
class SharedExport:
    modules: dict
    data: dict
    boxes: dict
    metadata: dict

    def savefig_kwargs(self, metric):
        # The Bbox already includes the padding. Never crop each panel again.
        return {"bbox_inches": self.boxes[metric], "pad_inches": 0.0}


def measure_content(fig, png_dpi):
    """Measure actual PDF and PNG text extents in original figure inches."""
    boxes = []
    axes_boxes = []

    def record(event):
        boxes.append(fig.get_tightbbox(event.renderer).frozen())
        axes_boxes.append(
            fig.axes[0].get_window_extent(event.renderer)
            .transformed(fig.dpi_scale_trans.inverted()).frozen()
        )

    connection = fig.canvas.mpl_connect("draw_event", record)
    try:
        # Save in memory to obtain each backend's real renderer. In particular,
        # PDF mathtext extents are not identical to Agg/PNG measurements.
        with plt.rc_context({"savefig.bbox": None}):
            for output_format in ("pdf", "png"):
                with io.BytesIO() as stream:
                    fig.savefig(stream, format=output_format, dpi=png_dpi,
                                bbox_inches=None)
    finally:
        fig.canvas.mpl_disconnect(connection)
    if len(boxes) != 2:
        raise RuntimeError("Expected one uncropped draw per PDF/PNG backend")
    return Bbox.union(boxes), axes_boxes


def prepare_shared_export(overrides=None):
    """Compute a fresh, per-metric union across all selected panels and sweeps.

    No on-disk or process-wide bbox cache: changed fonts, ticks, data selection
    and legend settings must be measured again before a new export batch.
    """
    overrides = overrides or {}
    modules = {
        family: overrides[family] if family in overrides else importlib.import_module(name)
        for family, name in MODULE_NAMES.items()
    }
    sample = modules["sample_rate"]
    base = sample.base
    sample.configure_fonts()
    rows, sources = base.load_all_rows()
    data = {
        "sample_rate": (base.build_scenarios(rows), sources),
        "bins": modules["bins"].load_bin_scenarios(),
        "reference_pool": modules["reference_pool"].load_k_scenarios(),
    }
    bounds = {}
    axes_by_metric = {}
    counts = {}
    for family, module in modules.items():
        scenarios, _ = data[family]
        for metric in module.METRICS_TO_PLOT:
            for dataset in base.DATASETS_TO_PLOT:
                for architecture in base.ARCHITECTURES_TO_PLOT:
                    keys = ([(dataset, architecture)] if family == "reference_pool"
                            else [(dataset, architecture, k) for k in base.K_VALUES_TO_PLOT])
                    for key in keys:
                        fig = module.create_panel(dataset, architecture, scenarios[key], metric)
                        try:
                            content_box, axes_boxes = measure_content(fig, module.PNG_DPI)
                            bounds.setdefault(metric, []).append(content_box)
                            counts[metric] = counts.get(metric, 0) + 1
                            for axes_box in axes_boxes:
                                previous = axes_by_metric.setdefault(metric, axes_box)
                                if not np.allclose(previous.extents, axes_box.extents,
                                                   rtol=0, atol=1e-9):
                                    raise ValueError(
                                        f"Axes geometry differs for {family}/{key}/{metric}; "
                                        "use identical figure sizes and panel margins."
                                    )
                        finally:
                            plt.close(fig)

    boxes = {metric: Bbox.union(items).padded(SHARED_PAD_INCHES)
             for metric, items in bounds.items()}
    metadata = {
        "mode": "shared_tight",
        "scope": "all selected datasets and architectures across all three sweeps, per metric",
        "measurement_backends": ["pdf", "png"],
        "padding_inches": SHARED_PAD_INCHES,
        "panels_measured_by_metric": counts,
        "bbox_extents_inches_by_metric": {
            metric: box.extents.tolist() for metric, box in boxes.items()
        },
        "page_size_points_by_metric": {
            metric: [box.width * 72, box.height * 72] for metric, box in boxes.items()
        },
        "axes_extents_points_by_metric": {
            metric: ((axes_by_metric[metric].extents -
                      np.tile(box.p0, 2)) * 72).tolist()
            for metric, box in boxes.items()
        },
        "render_scripts_sha256": {
            family: sample.sha256(module.__file__) for family, module in modules.items()
        },
        "shared_export_script_sha256": sample.sha256(__file__),
        "source_tables_by_family": {family: sources for family, (_, sources) in data.items()},
    }
    return SharedExport(modules, data, boxes, metadata)


def main():
    # A new directory per complete batch preserves all earlier figure exports.
    output_root = ROOT / "saved_plots" / (
        "unknown_arch_three_sweeps_shared_tight_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shared_export = prepare_shared_export()
    output_root.mkdir(parents=True, exist_ok=False)
    for family, module in shared_export.modules.items():
        module.OUTPUT_DIR = output_root / family
        module.main(shared_export=shared_export)

    legend = importlib.import_module("plot_unknown_arch_sweeps_shared_legend")
    legend.OUTPUT_DIR = output_root / "shared_legend"
    legend.main()
    (output_root / "shared_crop_manifest.json").write_text(
        json.dumps(shared_export.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Complete shared-tight batch: {output_root}")
    return output_root


if __name__ == "__main__":
    sys.modules.setdefault("plot_unknown_arch_sweeps_shared_tight", sys.modules[__name__])
    main()
