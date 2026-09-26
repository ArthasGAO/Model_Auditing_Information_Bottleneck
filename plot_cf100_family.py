"""CIFAR-100 family plots; reuse notebook selectors/style and archived H0 fits.

Seven methods x 50 seeds x three architectures. HL is illustrative synthetic
data from multiple1. KD/DKD and pruning preft are excluded. CSVs are read-only.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_fixed_split_h0 as fixed_h0

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "distribution_check.ipynb"
CSV = {
    "ft": ROOT / "saved_logs/ft_final/MI_master_table_ft_multiple.csv",
    "prune": ROOT / "saved_logs/pruning_final/MI_master_table_prune_multiple.csv",
    "knockoff": ROOT / "saved_logs/extraction_final/MI_master_table_extraction_multiple.csv",
    "hl": ROOT / "saved_logs/extraction_final/MI_master_table_extraction_multiple1.csv",
    "negative": ROOT / "saved_logs/vanilla/MI_master_table_neg_pool0.csv",
    "victim": ROOT / "saved_logs/vanilla/MI_master_table_victim.csv",
}
RESULT_DIR = ROOT / "saved_logs/vanilla/Hypo_Test_FixedSplits/In_rate1_bins50"
ARCHITECTURES = {
    "RN18": dict(anchor="CIFAR-100_ResNet-18_25000", extraction="ResNet-18", model="ResNet-18", suffix="Same18"),
    "VGG16": dict(anchor="CIFAR-100_VGG16_25000", extraction="VGG16", model="VGG16", suffix="Same16"),
    "DeiT": dict(anchor="CIFAR-100_DeiT_Distill_25000", extraction="DeiT",
                 model="deit_tiny_distilled_patch16_224", suffix="SameDeiT"),
}
FAMILIES = {"FT": ["FT-LL", "FT-AL", "RT-AL"], "Pruning": ["P-20%", "P-80%"],
            "Extraction": ["Knockoff", "HL"]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _namespace(include_framework=False):
    """Execute only definitions/configs; never execute old rendering cells."""
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module = types.ModuleType("_cf100_family_notebook")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    if include_framework:
        source = next("".join(c["source"]) for c in reversed(book["cells"])
                      if "class TableGroupSpec:" in "".join(c["source"]))
        exec(compile(source, "notebook-MI-framework", "exec"), ns)
    for identity in ("fixed-h0-single-config", "fixed-h0-overlay-config", "fixed-h0-family-config"):
        source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == identity)
        exec(compile(source, identity, "exec"), ns)
    return ns


def default_config():
    """Editable defaults copied from the existing CF10 family presentation."""
    ns = _namespace()
    palette = ns["FAMILY_METHOD_COLORS"]
    return dict(
        architectures=list(ARCHITECTURES), families=deepcopy(FAMILIES),
        bins=50, in_size=25000, k=30, round_id=44, alpha=.01,
        output_dir=ROOT / "saved_plots/Fig_CF100_SameArch_family_overlay",
        figsize=tuple(ns["FAMILY_FIGSIZE"]), font_family=ns["FAMILY_FONT_FAMILY"],
        font_sizes=deepcopy(ns["FAMILY_FONT_SIZES"]), role_styles=deepcopy(ns["FAMILY_ROLE_STYLES"]),
        method_colors={"FT-LL": palette["FT-LL"], "FT-AL": palette["FT-AL"], "RT-AL": palette["RT-AL"],
                       "P-20%": palette["PR20% best"], "P-80%": palette["PR80% best"],
                       "Knockoff": palette["Knockoff"], "HL": palette["DFMS"]},
        method_labels={}, legend_options=deepcopy(ns["FAMILY_LEGEND_OPTIONS"]),
        legend_labels=deepcopy(ns["FAMILY_LEGEND_LABELS"]), axes_options=deepcopy(ns["FAMILY_AXES_OPTIONS"]),
        boundary_style=deepcopy(ns["FAMILY_BOUNDARY_STYLE"]), reference_open=ns["FAMILY_REFERENCE_HOLLOW"],
        reference_marker=ns["FAMILY_REFERENCE_MARKER"], reference_area=ns["FAMILY_REFERENCE_RING_AREA"],
        reference_width=ns["FAMILY_REFERENCE_RING_WIDTH"], reference_zorder=ns["FAMILY_REFERENCE_RING_ZORDER"],
        shared_limits=ns["FAMILY_SHARED_LIMITS"], shared_crop=ns["FAMILY_SHARED_CROP"],
        pad_inches=ns["FAMILY_PAD_INCHES"], pdf_dpi=ns["FAMILY_PDF_DPI"], png_dpi=ns["FAMILY_PNG_DPI"],
        save_png=True, save_overview=True, show=False)


def _select(tables, arch, method, config):
    meta = ARCHITECTURES[arch]
    if method in FAMILIES["FT"]:
        kind, scenario = "ft", meta["anchor"] + "_Same_25000"
        extra = dict(strategy=method, ft_size=25000, model_seed=42)
    elif method in FAMILIES["Pruning"]:
        kind, scenario = "prune", meta["anchor"] + "_Same_25000"
        extra = dict(strategy="FT-AL", sparsity=.2 if method == "P-20%" else .8,
                     ckpt_kind="best", ft_size=25000, model_seed=42)
    else:
        kind = "knockoff" if method == "Knockoff" else "hl"
        middle = "Knockoff_Same100" if kind == "knockoff" else "DFMS_Illustrative"
        scenario = f'CIFAR-100_{meta["extraction"]}_25000_{middle}_{meta["suffix"]}'
        extra = dict(attack="Knockoff" if kind == "knockoff" else "DFMS",
                     victim_model=meta["model"], substitute_model=meta["model"],
                     epoch="best" if kind == "knockoff" else "synthetic")
    df = tables[kind]
    mask = ((df.Scenario == scenario) & (df.bins == config["bins"])
            & (df.in_size == config["in_size"]) & (df.rate == 1))
    for column, value in extra.items():
        mask &= df[column] == value
    chosen = df[mask].sort_values("seed").copy()
    require(chosen.seed.tolist() == list(range(50)) and chosen.model_name.is_unique,
            f"{arch} {method}: expected exactly seed0..49, found {len(chosen)} rows")
    require(np.isfinite(chosen[["I(X;T)-In", "I(T;Y)-In"]].to_numpy()).all(), f"{arch} {method}: nonfinite In MI")
    return kind, scenario, chosen


def _save_overview(png_paths, names, config):
    from PIL import Image
    rows, cols = len(config["architectures"]), len(config["families"])
    overview, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
    overview.subplots_adjust(left=.015, right=.995, bottom=.008, top=.97, wspace=.015, hspace=.06)
    for ax, image_path, name in zip(axes.flat, png_paths, names):
        with Image.open(image_path) as picture:
            ax.imshow(np.asarray(picture))
        ax.axis("off")
        ax.set_title(name.replace("CF100_", "").replace("_", " / "),
                     fontfamily=config["font_family"], fontsize=13, pad=2)
    destination = Path(config["output_dir"]) / "CF100_family_overview.png"
    overview.savefig(destination, dpi=160, bbox_inches="tight", pad_inches=.03, facecolor="white")
    plt.close(overview)
    return destination


def render(config=None):
    config = deepcopy(default_config() if config is None else config)
    require(config["bins"] == 50 and config["in_size"] == 25000, "Coverage is for bins=50, in_size=25000")
    arches = config["architectures"]
    require(arches and len(set(arches)) == len(arches) and set(arches) <= ARCHITECTURES.keys(), "Invalid architectures")
    requested = [m for family in config["families"].values() for m in family]
    require(requested and len(set(requested)) == len(requested), "Methods must appear in exactly one family")
    for family, methods in config["families"].items():
        require(family in FAMILIES and methods and set(methods) <= set(FAMILIES[family]), "Invalid family methods")
    protected = [*CSV.values(), ROOT / "saved_logs/vanilla/fixed_splits/pool0_80_models_v1.json",
                 *[RESULT_DIR / name for name in ("run_metadata.json", "per_split.csv", "per_model.csv")]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    ns = _namespace(include_framework=True)
    tables = {kind: pd.read_csv(CSV[kind]) for kind in ("ft", "prune", "knockoff", "hl")}
    selected, coverage = {}, []
    for arch in arches:
        for family, methods in config["families"].items():
            for method in methods:
                kind, scenario, rows = _select(tables, arch, method, config)
                selected[arch, method] = (kind, scenario, rows)
                coverage.append(dict(architecture=arch, family=family, method=method, n=len(rows),
                    scenario=scenario, source=str(CSV[kind].relative_to(ROOT)),
                    provenance="illustrative_synthetic" if method == "HL" else "measured_anchors_with_synthetic_completion"))
    nulls = fixed_h0.load_nulls([ARCHITECTURES[a]["anchor"] for a in arches],
        bins=config["bins"], in_size=config["in_size"], k=config["k"], round_id=config["round_id"],
        alpha=config["alpha"], negative_csv=CSV["negative"], result_dir=RESULT_DIR)
    figures, axes, names, panels, point_results = [], [], [], [], []
    try:
        for arch in arches:
            meta = ARCHITECTURES[arch]
            h0 = "c100_" + arch.lower()
            null = nulls[meta["anchor"]]
            for family, methods in config["families"].items():
                panel = dict(negative_groups={h0: dict(scenario=meta["anchor"], label=f"CF100 {arch}", seeds=None)},
                    positive_groups=[], victims=[ns["VictimSpec"](csv_path=str(CSV["victim"]), scenario=meta["anchor"],
                    seeds=[42], rates=[1.], bins=config["bins"], in_size=config["in_size"], domain="in")])
                for method in methods:
                    kind, scenario, data = selected[arch, method]
                    spec = ns["TableGroupSpec"](label=method, csv_path=str(CSV[kind]), scenario=scenario,
                        model_name=data.model_name.tolist(), rates=[1.], bins=config["bins"],
                        in_size=config["in_size"], domain="in", style={})
                    panel["positive_groups"].append(dict(h0=h0, synthetic=True, spec=spec))
                fig, ax, _, points = fixed_h0.plot_fixed_split_single(panel, framework=ns,
                    figsize=config["figsize"], font_family=config["font_family"], font_sizes=config["font_sizes"],
                    role_styles=config["role_styles"], legend_options=config["legend_options"],
                    legend_labels=config["legend_labels"], axes_options=config["axes_options"],
                    boundary_style=config["boundary_style"],
                    positive_group_colors={m: config["method_colors"][m] for m in methods},
                    positive_group_labels={m: config["method_labels"].get(m, m) for m in methods},
                    bins=config["bins"], in_size=config["in_size"], k=config["k"], round_id=config["round_id"],
                    alpha=config["alpha"], negative_csv=CSV["negative"], result_dir=RESULT_DIR, show=False)
                figures.append(fig)
                if config["reference_open"]:
                    fixed_h0.style_reference_open_marker(ax, marker=config["reference_marker"],
                        color=config["role_styles"]["reference_style"]["color"], area=config["reference_area"],
                        linewidth=config["reference_width"], zorder=config["reference_zorder"],
                        legend_label=config["legend_labels"]["reference_style"])
                # Check actual coordinates/identities and boundary, not just counts.
                np.testing.assert_array_equal(ax.collections[0].get_offsets(), null.evaluation[["I(X;T)-In", "I(T;Y)-In"]])
                np.testing.assert_array_equal(ax.collections[1].get_offsets(), null.reference[["I(X;T)-In", "I(T;Y)-In"]])
                np.testing.assert_allclose(ax.lines[0].get_xydata(), null.boundary(), rtol=0, atol=0)
                for method in methods:
                    _, _, data = selected[arch, method]
                    actual = points[(points.role == "positive") & (points.group == method)].sort_values("seed")
                    require(actual.seed.tolist() == list(range(50)), f"Incomplete plotted seeds: {arch} {method}")
                    require(actual.model_name.tolist() == data.model_name.tolist(), "Plotted identity mismatch")
                    np.testing.assert_array_equal(actual[["ixt", "ity"]], data[["I(X;T)-In", "I(T;Y)-In"]])
                fig.canvas.draw()
                require(np.isclose(ax.bbox.width, ax.bbox.height), "Plotting area is not square")
                axes.append(ax)
                names.append(f"CF100_{arch}_{family}")
                panels.append(panel)
                point_results.append(points.assign(architecture=arch, method_family=family))
                print(f"{names[-1]}: {', '.join(methods)} ({50*len(methods)} positive seeds)")
        if config["shared_limits"]:
            count = len(config["families"])
            for offset in range(0, len(axes), count):
                group = axes[offset:offset+count]
                xlim = (min(a.get_xlim()[0] for a in group), max(a.get_xlim()[1] for a in group))
                ylim = (min(a.get_ylim()[0] for a in group), max(a.get_ylim()[1] for a in group))
                for ax in group:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
        out = Path(config["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        pdfs = [out / f"{name}.pdf" for name in names]
        pngs = [out / f"{name}.png" for name in names]
        fixed_h0.save_panel_figures(figures, pdfs, dpi=config["pdf_dpi"],
            pad_inches=config["pad_inches"], shared_crop=config["shared_crop"])
        if config["save_png"] or config["save_overview"]:
            fixed_h0.save_panel_figures(figures, pngs, dpi=config["png_dpi"],
                pad_inches=config["pad_inches"], shared_crop=config["shared_crop"])
        overview = _save_overview(pngs, names, config) if config["save_overview"] else None
        require(all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items()),
                "A source CSV or archived test file changed during rendering")
        manifest = dict(dataset="CIFAR-100", bins=config["bins"], in_size=config["in_size"],
            round_id=config["round_id"], k=config["k"], alpha=config["alpha"],
            skipped=["KD", "DKD", "pruning preft"], coverage=coverage, source_hashes=hashes,
            figure_names=names, all_plotted_coordinates_verified=True,
            note="HL groups are entirely illustrative synthetic MI; other positive clouds include synthetic completion.")
        (out / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if config["show"]:
            plt.show()
        else:
            for fig in figures:
                plt.close(fig)
        print(f"Saved {len(pdfs)} family PDFs to {out}")
        return dict(figures=figures, axes=axes, panels=panels, names=names, coverage=pd.DataFrame(coverage),
                    points=pd.concat(point_results, ignore_index=True), pdf_paths=pdfs, png_paths=pngs,
                    overview=overview, manifest=manifest)
    except Exception:
        for fig in figures:
            plt.close(fig)
        raise


if __name__ == "__main__":
    render()
