"""F-test ellipses for the notebook's existing TableGroupSpec plotting API.

No fitting on evaluation points, no seed sampling, no MI or result-file writes.
The archived per-split fit is checked against current reference coordinates.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import f

from run_hypothesis_test_fixed_splits import load_manifest, get_split, score, digest

BASE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass
class FixedNull:
    scenario: str
    k: int
    alpha: float
    mu: np.ndarray
    covariance: np.ndarray
    reference: pd.DataFrame
    evaluation: pd.DataFrame
    cutoff_md2: float

    def p_values(self, xy):
        """Reuse the experiment's authoritative predictive F implementation."""
        return score(self.reference[["I(X;T)-In", "I(T;Y)-In"]].to_numpy(), xy)[4]

    def boundary(self, n=361):
        angle = np.linspace(0, 2*np.pi, n)
        circle = np.column_stack((np.cos(angle), np.sin(angle)))
        return self.mu + np.sqrt(self.cutoff_md2) * circle @ np.linalg.cholesky(self.covariance).T


def load_nulls(scenarios, *, bins=50, in_size=25000, k=30, round_id=44, alpha=.01,
               negative_csv=BASE/"saved_logs/vanilla/MI_master_table_neg_pool0.csv",
               result_dir=BASE/"saved_logs/vanilla/Hypo_Test_FixedSplits/In_rate1_bins50",
               manifest=BASE/"saved_logs/vanilla/fixed_splits/pool0_80_models_v1.json"):
    require(0 < alpha < 1, "alpha must be between 0 and 1")
    doc = load_manifest(Path(manifest))
    result_dir = Path(result_dir)
    meta = json.loads((result_dir/"run_metadata.json").read_text())
    require(meta["status"] == "complete" and not meta["gate1"], "Need a complete, ungated F-test run")
    require(meta["mi_kind"] == "In" and meta["bins"] == bins, "Results use a different MI domain/bins")
    require(meta["manifest_sha256"] == doc["manifest_sha256"], "Split manifest mismatch")
    require(meta["mi_csv_sha256"] == digest(Path(negative_csv).read_bytes()),
            "Negative MI CSV has changed since testing; rerun the affected hypothesis tests first")
    neg = pd.read_csv(negative_csv)
    neg = neg[(neg.bins == bins) & (neg.in_size == in_size) & (neg.rate == 0)]
    saved = pd.read_csv(result_dir/"per_split.csv")
    saved_models = pd.read_csv(result_dir/"per_model.csv")
    nulls = {}
    for scenario in scenarios:
        require(meta["in_sizes_by_case"].get(scenario) == in_size, "Result in_size does not match")
        split = get_split(doc, scenario, round_id, k)
        pool = neg[neg.Scenario == scenario].copy()
        require(len(pool) == 80 and pool.model_name.is_unique and pool.seed.is_unique,
                f"{scenario}: need exactly 80 unique pool models")
        require(np.isfinite(pool[["I(X;T)-In", "I(T;Y)-In"]].to_numpy()).all(), "Nonfinite MI")
        def select(which):
            names = [m["model_name"] for m in split[which]]
            result = pool.set_index("model_name", drop=False).loc[names].reset_index(drop=True)
            require(result.seed.tolist() == [m["seed"] for m in split[which]], "Model/seed mismatch")
            return result
        ref, ev = select("h0"), select("evaluation_negative")
        cols = ["I(X;T)-In", "I(T;Y)-In"]
        mu, cov, _, _, pf = score(ref[cols].to_numpy(), ev[cols].to_numpy())
        record = saved[(saved.scenario == scenario) & (saved.round_id == round_id) &
                       (saved.k_ref == k) & (saved.bins == bins) & (saved.in_size == in_size)]
        require(len(record) == 1, "Missing/duplicate saved split")
        r = record.iloc[0]
        require(json.loads(r.h0_seeds) == ref.seed.tolist() and
                json.loads(r.eval_negative_seeds) == ev.seed.tolist(), "Saved split identity mismatch")
        np.testing.assert_allclose(mu, json.loads(r.mu), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov, json.loads(r.covariance), rtol=1e-10, atol=1e-14)
        old = saved_models[(saved_models.scenario == scenario) & (saved_models.round_id == round_id) &
                           (saved_models.k_ref == k)]
        require(len(old) == 50 and old.model_name.is_unique, "Invalid saved evaluated negatives")
        old = old.set_index("model_name").loc[ev.model_name]
        np.testing.assert_allclose(old[["ixt", "ity"]], ev[cols], rtol=0, atol=1e-12)
        np.testing.assert_allclose(old.p_F, pf, rtol=1e-10, atol=1e-14)
        if f"nfp_F@{alpha}" in r:
            require(int(sum(pf < alpha)) == int(r[f"nfp_F@{alpha}"]), "Saved FPR mismatch")
        # Predictive T2 = k/(k+1) MD2; F = (k-2)/(2(k-1)) T2.
        cutoff = (k+1)/k * 2*(k-1)/(k-2) * f.ppf(1-alpha, 2, k-2)
        nulls[scenario] = FixedNull(scenario, k, alpha, mu, cov, ref, ev, float(cutoff))
    return nulls


def selected_points(spec, framework, bins, in_size):
    require(spec.domain == "in" and spec.bins == bins and spec.in_size == in_size,
            f"{spec.label}: must use the same In MI, bins and in_size as H0")
    require(spec.mode == "points", "Use points mode so every classified point is displayed")
    points = []
    for bucket in framework["points_from_tablegroupspec"](spec).values():
        points.extend(spec.select_fn(bucket) if spec.select_fn else bucket)
    require(bool(points), f"{spec.label}: no matching points; check Scenario/model_name/seeds")
    identities = [(p["model_name"], p["epoch"], p["seed"], p["rate"]) for p in points]
    require(len(set(identities)) == len(points), f"{spec.label}: duplicate selected rows")
    xy = np.array([[p["ix"], p["iy"]] for p in points])
    require(np.isfinite(xy).all(), f"{spec.label}: nonfinite MI")
    return points, xy


def check_knockoff_mapping(points, scenario):
    """Reject common wrong-dataset or wrong-suspect-pool assignments."""
    for p in points:
        name = str(p["scenario"])
        if "_Knockoff_" not in name:
            continue  # Other attack families explicitly select their H0 in config.
        require(name.split("_")[0] == scenario.split("_")[0], "Positive/H0 dataset mismatch")
        suffix = name.rsplit("_", 1)[-1]
        arch = {"Same18": "ResNet-18", "Cross18": "ResNet-18",
                "Same16": "VGG16", "Cross16": "VGG16",
                "SameDeiT": "DeiT", "CrossDeiT": "DeiT"}.get(suffix)
        require(arch is not None and f"_{arch}" in scenario, f"{name}: wrong suspect-architecture H0 {scenario}")


def _format_square_figure(fig, axes, *, font_family, handles):
    """Fixed canvas/axes geometry; do not let legends or tight_layout resize it."""
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    findfont(FontProperties(family=font_family), fallback_to_default=False)
    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_title("")
        ax.set_xlabel("I (X;T)", fontsize=13, fontfamily=font_family)
        ax.set_ylabel("I (T;Y)", fontsize=13, fontfamily=font_family)
        ax.tick_params(labelsize=11)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(alpha=.25)
        ax.legend(handles=handles, loc="best", frameon=True,
                  prop=FontProperties(family=font_family, size=10))
    # Notebook inline rendering often uses bbox_inches='tight'. Include the
    # entire square canvas in that bounding box, without any visible border.
    frame = Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                      facecolor="none", edgecolor="none", linewidth=0)
    frame.set_in_layout(True)
    fig.add_artist(frame)
    fig.canvas.draw()
    for text in fig.findobj(Text):
        text.set_fontfamily(font_family)
    fig.canvas.draw()


def save_square_figure(fig, path, dpi=300):
    """Export the exact square canvas, overriding any global tight-crop setting."""
    import matplotlib.pyplot as plt
    width, height = fig.get_size_inches()
    require(np.isclose(width, height), "Figure canvas must be square")
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(path, dpi=dpi, bbox_inches=None)


def plot_fixed_split_plane(negative_groups, positive_groups, victims, *, framework,
                          bins=50, in_size=25000, k=30, round_id=44, alpha=.01,
                          negative_csv=BASE/"saved_logs/vanilla/MI_master_table_neg_pool0.csv",
                          result_dir=BASE/"saved_logs/vanilla/Hypo_Test_FixedSplits/In_rate1_bins50",
                          show_reference=False, zoom=True, title=None,
                          figsize=(6, 6), show=True, suspect_style=None,
                          reference_style=None, victim_style=None,
                          positive_style=None, negative_style=None,
                          font_family="Times New Roman", target_ax=None):
    """Return (fig, ax, summary_df, point_df, zoom_fig), without saving files.

    negative_groups: key -> {scenario, label, color, seeds=None, style={}}.
      seeds=None plots exactly the frozen 50 evaluated negatives; a custom seed
      list must be a subset of those 50. It never changes H0 or reported full FPR.
    positive_groups: list of {spec: TableGroupSpec, h0: negative key,
                              synthetic: bool}. Filters/styles remain editable.
    framework: notebook globals containing the existing plotting framework.
    Role styles apply to both main and zoom plots. Reference points are shown
    separately and never enter evaluation counts or the suspect-point table.
    positive_style and negative_style are independent. The legacy suspect_style
    is only a shared fallback; explicit role styles take precedence over it.
    Figures and axes are square, with a role-only internal legend and no title
    or footer. title is accepted only for compatibility and is not displayed.
    target_ax embeds the main plot into a caller-owned layout (no local legend
    or formatting); zoom must then be False.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    require(len(figsize) == 2 and min(figsize) > 0 and np.isclose(*figsize),
            "FIGSIZE must be square, e.g. (6, 6)")
    require(target_ax is None or not zoom, "Embedded panels require zoom=False")
    positive_style = {**(suspect_style or {}), **(positive_style or {})}
    negative_style = {**(suspect_style or {}), **(negative_style or {})}
    reference_style = {"marker": "o", "color": "#1f77b4", "s": 42,
                       "alpha": 1.0, "edgecolors": "none", "linewidths": 0, "zorder": 2,
                       **(reference_style or {})}
    require(bool(negative_groups), "Choose at least one negative/H0 group")
    required = ["TableGroupSpec", "MemorySpec", "plot_information_plane",
                "points_from_tablegroupspec", "clear_csv_cache", "_resolve_victim_point"]
    require(all(name in framework for name in required),
            "Run the notebook's MI information-plane plotting framework cell first")
    framework["clear_csv_cache"]()
    scenarios = [g["scenario"] for g in negative_groups.values()]
    require(len(set(scenarios)) == len(scenarios), "Each negative pool should appear only once")
    nulls = load_nulls(scenarios, bins=bins, in_size=in_size, k=k, round_id=round_id,
                      alpha=alpha, negative_csv=negative_csv, result_dir=result_dir)
    groups, report, details, curves, negative_scatter = [], [], [], {}, {}
    labels = []
    for key, config in negative_groups.items():
        null = nulls[config["scenario"]]
        label, color = config.get("label", key), config.get("color", "#777777")
        full = null.evaluation
        seeds = config.get("seeds")
        if seeds is not None:
            require(len(seeds) > 0 and len(set(seeds)) == len(seeds) and set(seeds) <= set(full.seed),
                    f"{key}: display seeds must be unique evaluated-negative seeds, never reference seeds")
        ev = full if seeds is None else full[full.seed.isin(seeds)]
        xy = ev[["I(X;T)-In", "I(T;Y)-In"]].to_numpy()
        full_pf = null.p_values(full[["I(X;T)-In", "I(T;Y)-In"]].to_numpy())
        pf = null.p_values(xy)
        fp = int(sum(full_pf < alpha))
        evaluated_style = {"marker": "^", "color": "green", "s": 65, "alpha": .85,
                           "edgecolors": "none", "linewidths": 0, "zorder": 3,
                           **config.get("style", {}), **negative_style}
        groups.append(framework["MemorySpec"](
            label=f"{label}: evaluated negatives n={len(ev)}; full FP={fp}/50",
            points=[tuple(x) for x in xy],
            style=evaluated_style))
        if show_reference:
            groups.append(framework["MemorySpec"](
                label=f"{label}: H0 reference n={k}",
                points=[tuple(x) for x in null.reference[["I(X;T)-In", "I(T;Y)-In"]].to_numpy()],
                style=reference_style))
        report.append(dict(group=label, role="negative", h0=key, n=len(ev),
                           outside=int(sum(pf < alpha)), inside=int(sum(pf >= alpha)),
                           FPR=float(np.mean(pf < alpha)), FNR=np.nan,
                           full_evaluation_n=50, full_evaluation_FPR=fp/50, synthetic=False))
        for row, prob in zip(ev.to_dict("records"), pf):
            details.append(dict(group=label, role="negative", h0=key, model_name=row["model_name"],
                                seed=row["seed"], ixt=row["I(X;T)-In"], ity=row["I(T;Y)-In"],
                                p_F=float(prob), outside=bool(prob < alpha)))
        curves[key] = (null.boundary(), color, label)
        negative_scatter[key] = (xy, pf, evaluated_style)
        labels.append(label)
    positive_scatter = []
    for config in positive_groups:
        spec, key = config["spec"], config["h0"]
        require(key in negative_groups, f"Unknown H0 group {key!r}; add it to NEGATIVE_GROUPS")
        null = nulls[negative_groups[key]["scenario"]]
        points, xy = selected_points(spec, framework, bins, in_size)
        check_knockoff_mapping(points, null.scenario)
        pf = null.p_values(xy)
        inside = int(sum(pf >= alpha))
        synthetic = bool(config.get("synthetic", "_multiple" in str(spec.csv_path)))
        plotted_style = {**spec.style, **positive_style}
        groups.append(replace(spec, label=f"{spec.label}: n={len(points)}, inside={inside}",
                              style=plotted_style))
        report.append(dict(group=spec.label, role="positive", h0=key, n=len(points),
                           outside=len(points)-inside, inside=inside, FPR=np.nan,
                           FNR=inside/len(points), full_evaluation_n=np.nan,
                           full_evaluation_FPR=np.nan, synthetic=synthetic))
        for pt, prob in zip(points, pf):
            details.append(dict(group=spec.label, role="positive", h0=key, model_name=pt["model_name"],
                                seed=pt["seed"], ixt=pt["ix"], ity=pt["iy"],
                                p_F=float(prob), outside=bool(prob < alpha)))
        positive_scatter.append((key, xy, plotted_style))
        labels.append(spec.label)
    require(len(labels) == len(set(labels)), "Group labels must be unique")
    victims = [replace(v, style={**(v.style or {}), **(victim_style or {})}) for v in victims]
    for v in victims:
        require(v.domain == "in" and v.bins == bins and v.in_size == in_size, "Victim uses a different grid")
        framework["_resolve_victim_point"](v)  # Validate before creating a figure.
    # Keep the existing group selectors, but own layout here: the older renderer
    # reserves space for verbose per-group legends via tight_layout.
    if target_ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout=None)
        fig.set_layout_engine(None)
        fig.subplots_adjust(left=.16, right=.96, bottom=.14, top=.94)
    else:
        ax, fig = target_ax, target_ax.figure
    for group in groups:
        if isinstance(group, framework["MemorySpec"]):
            xy = np.asarray(group.points)
        else:
            _, xy = selected_points(group, framework, bins, in_size)
        ax.scatter(*xy.T, **{**group.style, "edgecolors": "none", "linewidths": 0})
    for victim in victims:
        point = framework["_resolve_victim_point"](victim)
        ax.scatter(point["ix"], point["iy"], **{**victim.style, "edgecolors": "none", "linewidths": 0})
    for curve, color, label in curves.values():
        ax.plot(*curve.T, color=color, linestyle="--", linewidth=1.8)
    def legend_marker(label, style, default_marker, default_color):
        return Line2D([], [], linestyle="none", label=label,
                      marker=style.get("marker", default_marker),
                      markerfacecolor=style.get("color", default_color),
                      markeredgecolor="none", markeredgewidth=0, markersize=7)
    handles = []
    if victims:
        handles.append(legend_marker("Victim model", victims[0].style, "s", "black"))
    if positive_scatter:
        handles.append(legend_marker("positive suspects", positive_scatter[0][2], "^", "red"))
    handles.append(legend_marker("negative suspects", next(iter(negative_scatter.values()))[2], "^", "green"))
    if show_reference:
        handles.append(legend_marker("reference models", reference_style, "o", "#1f77b4"))
    if target_ax is None:
        _format_square_figure(fig, [ax], font_family=font_family, handles=handles)
    zoom_fig = None
    if zoom:
        side = int(np.ceil(np.sqrt(len(curves))))
        zoom_fig, axes = plt.subplots(side, side, figsize=figsize, squeeze=False, layout=None)
        zoom_fig.set_layout_engine(None)
        if side == 1:
            zoom_fig.subplots_adjust(left=.16, right=.96, bottom=.14, top=.94)
        else:
            zoom_fig.subplots_adjust(left=.12, right=.97, bottom=.12, top=.97, wspace=.55, hspace=.55)
        used_axes = []
        for axz, (key, (curve, color, label)) in zip(axes.flat, curves.items()):
            xy, pf, evaluated_style = negative_scatter[key]
            axz.plot(*curve.T, color=color, linestyle="--", linewidth=1.8)
            axz.scatter(*xy.T, **{**evaluated_style, "edgecolors": "none", "linewidths": 0})
            rejected = xy[pf < alpha]
            if len(rejected):
                axz.scatter(*rejected.T, marker="^", color="red", edgecolors="none", linewidths=0, s=100)
            extent = np.vstack((curve, xy))
            if show_reference:
                ref = nulls[negative_groups[key]["scenario"]].reference
                axz.scatter(ref["I(X;T)-In"], ref["I(T;Y)-In"],
                            **{**reference_style, "edgecolors": "none", "linewidths": 0})
                extent = np.vstack((extent, ref[["I(X;T)-In", "I(T;Y)-In"]].to_numpy()))
            for positive_key, points, style in positive_scatter:
                if positive_key == key:
                    axz.scatter(*points.T, **{**style, "edgecolors": "none", "linewidths": 0})
            low, high = extent.min(0), extent.max(0)
            pad = np.maximum(high-low, 1e-6)*.12
            axz.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]),
                    xlabel="I (X;T)", ylabel="I (T;Y)")
            used_axes.append(axz)
        for unused in list(axes.flat)[len(curves):]:
            unused.remove()
        zoom_handles = [h for h in handles if h.get_label() in ("negative suspects", "reference models")]
        _format_square_figure(zoom_fig, used_axes, font_family=font_family, handles=zoom_handles)
    if show:
        plt.show()
    return fig, ax, pd.DataFrame(report), pd.DataFrame(details), zoom_fig


def plot_fixed_split_grid(panels, *, framework, width_inches=6.5,
                         font_family="Times New Roman", tick_size=7,
                         label_size=8, legend_size=8, panel_captions=None,
                         caption_size=8, caption_wrap_chars=22,
                         caption_gap_inches=.08, row_gap_inches=.45, marker_scale=1.0,
                         boundary_width=1.2, boundary_color="#444444",
                         show_negative_inset=False, inset_bounds=(.08, .38, .50, .50),
                         inset_marker_scale=.65,
                         show=True, **test_options):
    """Two rows x four square panels, with one shared role legend.

    Each panel independently supplies negative_groups, positive_groups, victims.
    test_options contains the unchanged test configuration, e.g. bins/k/alpha.
    The physical width is the intended final paper width, not a preview size.
    panel_captions is eight editable strings in row-major order; defaults to
    (a)..(h). Explicit newlines are preserved, long lines wrap, and row/bottom
    space grows with the longest caption without shrinking the square axes.
    row_gap_inches is the inter-row reserve before adding caption space. A
    minimum clearance is retained below captions even if it is set too low.
    marker_scale multiplies marker linear dimensions (scatter area uses its
    square), including the shared legend. Coordinates and H0 stay unchanged.
    Optional insets magnify the same data and exact F boundary, not a dilated
    acceptance region. Their box/connector identifies the main-plot source.
    Returns fig, axes[2,4], summary table, point table. No data are generated.
    """
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    require(len(panels) == 8, "The 2 x 4 layout requires exactly eight panels")
    require(np.isfinite(marker_scale) and marker_scale > 0,
            "marker_scale must be finite and positive")
    require(np.isfinite(boundary_width) and boundary_width > 0,
            "boundary_width must be finite and positive")
    require(np.isfinite(inset_marker_scale) and inset_marker_scale > 0,
            "inset_marker_scale must be finite and positive")
    require(len(inset_bounds) == 4 and all(np.isfinite(inset_bounds))
            and min(inset_bounds[:2]) >= 0 and min(inset_bounds[2:]) > 0
            and inset_bounds[0]+inset_bounds[2] <= 1
            and inset_bounds[1]+inset_bounds[3] <= 1,
            "inset_bounds must fit inside the main axes")
    require(np.isfinite(width_inches) and width_inches >= 5.5,
            "Four panels need at least 5.5 inches here; use fewer columns for a narrow paper column")
    if panel_captions is None:
        panel_captions = [f"({chr(97+i)})" for i in range(8)]
    require(not isinstance(panel_captions, str) and len(panel_captions) == 8
            and all(isinstance(c, str) for c in panel_captions),
            "panel_captions must contain exactly eight strings")
    require(np.isfinite(caption_size) and caption_size > 0
            and np.isfinite(caption_gap_inches) and caption_gap_inches >= 0,
            "Caption size must be positive and gap nonnegative")
    require(isinstance(caption_wrap_chars, int) and caption_wrap_chars > 0,
            "caption_wrap_chars must be a positive integer")
    require(np.isfinite(row_gap_inches) and row_gap_inches >= 0,
            "row_gap_inches must be finite and nonnegative")
    captions = ["\n".join(wrapped for line in c.split("\n")
                          for wrapped in (textwrap.wrap(line, width=caption_wrap_chars) or [""]))
                for c in panel_captions]
    max_lines = max((len(c.splitlines()) for c in captions if c), default=0)
    findfont(FontProperties(family=font_family), fallback_to_default=False)
    left, right, bottom, top, col_gap = .42, .10, .40, .42, .30
    extra = max_lines*caption_size*1.25/72 + caption_gap_inches if max_lines else 0
    bottom += extra
    # Top-row captions start .18 inches below the axes plus caption_gap_inches.
    # Keep .06 inches below their reserved height before the next row starts.
    row_gap = max(row_gap_inches, .24) + extra
    side = (width_inches-left-right-3*col_gap)/4
    height = 2*side+row_gap+bottom+top
    fig = plt.figure(figsize=(width_inches, height), layout=None)
    fig.set_layout_engine(None)
    axes = np.empty((2, 4), dtype=object)
    summaries, points = [], []
    styles = dict(
        victim_style=dict(marker="s", color="black", s=22, edgecolors="none", linewidths=0, zorder=5),
        positive_style=dict(marker="^", color="red", s=8, alpha=.85, edgecolors="none", linewidths=0, zorder=3),
        negative_style=dict(marker="^", color="green", s=8, alpha=.85, edgecolors="none", linewidths=0, zorder=3),
        reference_style=dict(marker="o", color="#1f77b4", s=5, alpha=1., edgecolors="none", linewidths=0, zorder=2))
    try:
        for index, panel in enumerate(panels):
            row, col = divmod(index, 4)
            ax = fig.add_axes([(left+col*(side+col_gap))/width_inches,
                               (bottom+(1-row)*(side+row_gap))/height,
                               side/width_inches, side/height])
            axes[row, col] = ax
            options = {**test_options, **styles, **panel.get("options", {})}
            for role, default_style in styles.items():
                scaled = {**default_style, **options[role]}
                scaled["s"] *= marker_scale**2
                scaled["linewidths"] = scaled.get("linewidths", 0)*marker_scale
                options[role] = scaled
            options.update(show_reference=True, zoom=False, show=False,
                           target_ax=ax, font_family=font_family)
            _, _, summary, detail, _ = plot_fixed_split_plane(
                panel["negative_groups"], panel["positive_groups"], panel["victims"],
                framework=framework, **options)
            summaries.append(summary.assign(panel=index+1))
            points.append(detail.assign(panel=index+1))
            ax.set_box_aspect(1)
            ax.set_title("")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            for axis in (ax.xaxis, ax.yaxis):
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                axis.set_major_formatter(formatter)
            ax.tick_params(labelsize=tick_size, length=2.5, width=.5, pad=2)
            ax.set_xlabel("I (X;T)" if row == 1 else "", fontsize=label_size, labelpad=2)
            ax.set_ylabel("I (T;Y)" if col == 0 else "", fontsize=label_size, labelpad=3)
            ax.grid(alpha=.2, linewidth=.4)
            ax.margins(.09)
            for spine in ax.spines.values():
                spine.set_linewidth(.6)
            for line in ax.lines:
                line.set_linewidth(boundary_width)
                line.set_color(boundary_color)
                line.set_zorder(4)
            if show_negative_inset:
                # First 2*N collections are evaluated negatives and references.
                # Use their original offsets and exact boundary vertices; never
                # estimate a new ellipse or change its statistical threshold.
                negative_collections = ax.collections[:2*len(panel["negative_groups"])]
                extent = np.vstack([line.get_xydata() for line in ax.lines]
                                   + [np.asarray(c.get_offsets()) for c in negative_collections])
                low, high = extent.min(0), extent.max(0)
                pad = np.maximum(high-low, 1e-6)*.15
                inset = ax.inset_axes(inset_bounds, zorder=6)
                inset.set_gid(f"negative-inset-{index+1}")
                for collection in ax.collections:
                    copied = inset.scatter(*np.asarray(collection.get_offsets()).T,
                                           s=collection.get_sizes()*inset_marker_scale**2,
                                           facecolors=collection.get_facecolors(),
                                           edgecolors=collection.get_edgecolors(),
                                           linewidths=collection.get_linewidths()*inset_marker_scale,
                                           zorder=collection.get_zorder())
                    copied.set_paths(collection.get_paths())
                for line in ax.lines:
                    inset.plot(*line.get_xydata().T, color=boundary_color,
                               linestyle="--", linewidth=boundary_width, zorder=4)
                inset.set(xlim=(low[0]-pad[0], high[0]+pad[0]),
                          ylim=(low[1]-pad[1], high[1]+pad[1]), xticks=[], yticks=[])
                for spine in inset.spines.values():
                    spine.set_color(".6")
                    spine.set_linewidth(.5)
                inset.text(.04, .97, "zoom", transform=inset.transAxes,
                           ha="left", va="top", fontsize=5.5, fontfamily=font_family, color=".35")
                ax.indicate_inset_zoom(inset, edgecolor=".5", linewidth=.6, alpha=.7, zorder=1)
            if captions[index]:
                axes_bottom = bottom+(1-row)*(side+row_gap)
                offset = (.34 if row == 1 else .18) + caption_gap_inches
                fig.text((left+col*(side+col_gap)+side/2)/width_inches,
                         (axes_bottom-offset)/height, captions[index],
                         ha="center", va="top", fontsize=caption_size,
                         fontfamily=font_family, linespacing=1.25,
                         gid=f"panel-caption-{index+1}")
        handles = [Line2D([], [], linestyle="none", marker=marker, color=color,
                          markeredgecolor="none", markeredgewidth=0,
                          markersize=4.5*marker_scale, label=label)
                   for marker, color, label in [
                       ("s", "black", "Victim model"), ("^", "red", "positive suspects"),
                       ("^", "green", "negative suspects"), ("o", "#1f77b4", "reference models")]]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
                   ncol=4, frameon=False, handletextpad=.4, columnspacing=1.2,
                   prop=FontProperties(family=font_family, size=legend_size))
        # Do not add an invisible full-canvas artist: it defeats tight cropping.
        fig.canvas.draw()
        for text in fig.findobj(Text):
            text.set_fontfamily(font_family)
        fig.canvas.draw()
    except Exception:
        plt.close(fig)
        raise
    if show:
        plt.show()
    return fig, axes, pd.concat(summaries, ignore_index=True), pd.concat(points, ignore_index=True)


def plot_fixed_split_single(panel, *, framework, figsize=(2.6, 2.6),
                            font_family="Times New Roman", font_sizes=None,
                            role_styles=None, legend_options=None, legend_labels=None,
                            axes_options=None, boundary_style=None, show=False,
                            positive_group_colors=None, positive_group_labels=None,
                            **test_options):
    """Independent square panel with an internal legend and no title/caption.

    Uses the exact same selectors and fixed-H0 computation as the grid. All
    presentation dictionaries can be configured in the notebook. Only roles
    present in this panel appear in its legend (DKD has no positive points).
    Optional positive_group_colors maps original spec labels to face colors and
    enables one legend entry per method; positive_group_labels changes only its
    display text. With neither mapping, the existing role-only legend is kept.
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    require(len(figsize) == 2 and np.isfinite(figsize).all()
            and min(figsize) > 0 and np.isclose(*figsize), "figsize must be square")
    findfont(FontProperties(family=font_family), fallback_to_default=False)
    fonts = {"xlabel": 10, "ylabel": 10, "xtick": 8, "ytick": 8, "legend": 7,
             **(font_sizes or {})}
    styles = dict(
        victim_style=dict(marker="s", color="black", s=38, edgecolors="none", linewidths=0, zorder=5),
        positive_style=dict(marker="^", color="red", s=18, alpha=.85, edgecolors="none", linewidths=0, zorder=3),
        negative_style=dict(marker="^", color="green", s=18, alpha=.85, edgecolors="none", linewidths=0, zorder=3),
        reference_style=dict(marker="o", color="#1f77b4", s=12, alpha=1., edgecolors="none", linewidths=0, zorder=2))
    for key, overrides in (role_styles or {}).items():
        require(key in styles, f"Unknown role style: {key}")
        styles[key].update(overrides)
    # Marker strokes are disabled across all presentation variants, including
    # legends. This also neutralizes stale outlined styles in notebook kernels.
    for style in styles.values():
        style.update(edgecolors="none", linewidths=0)
    legend = dict(loc="best", frameon=True, framealpha=.9, facecolor="white",
                  edgecolor=".75", borderpad=.35, labelspacing=.3, handletextpad=.4,
                  handlelength=1., borderaxespad=.4)
    legend.update(legend_options or {})
    labels = {"victim_style": "Victim model", "positive_style": "positive suspects",
              "negative_style": "negative suspects", "reference_style": "reference models",
              **(legend_labels or {})}
    appearance = dict(rect=(.20, .18, .77, .77), xlabel="I (X;T)", ylabel="I (T;Y)",
                      xlabelpad=2, ylabelpad=3, tick_length=2.5, tick_width=.5, tick_pad=2,
                      x_nbins=3, y_nbins=3, data_margin=.09, spine_width=.6,
                      grid=True, grid_alpha=.2, grid_width=.4)
    appearance.update(axes_options or {})
    edge = dict(color="#444444", linewidth=1.2, linestyle="--", zorder=4)
    edge.update(boundary_style or {})
    fig = plt.figure(figsize=figsize, layout=None)
    fig.set_layout_engine(None)
    ax = fig.add_axes(appearance["rect"])
    try:
        options = {**test_options, **panel.get("options", {}), **styles}
        options.update(show_reference=True, zoom=False, show=False,
                       target_ax=ax, font_family=font_family)
        _, _, summary, points, _ = plot_fixed_split_plane(
            panel["negative_groups"], panel["positive_groups"], panel["victims"],
            framework=framework, **options)
        ax.set_box_aspect(1)
        ax.set_title("")
        ax.set_xlabel(appearance["xlabel"], fontsize=fonts["xlabel"], labelpad=appearance["xlabelpad"])
        ax.set_ylabel(appearance["ylabel"], fontsize=fonts["ylabel"], labelpad=appearance["ylabelpad"])
        for name in ("x", "y"):
            axis = getattr(ax, name+"axis")
            axis.set_major_locator(MaxNLocator(nbins=appearance[name+"_nbins"], min_n_ticks=2))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
            ax.tick_params(axis=name, labelsize=fonts[name+"tick"], length=appearance["tick_length"],
                           width=appearance["tick_width"], pad=appearance["tick_pad"])
        ax.margins(appearance["data_margin"])
        if appearance["grid"]:
            ax.grid(True, alpha=appearance["grid_alpha"], linewidth=appearance["grid_width"])
        else:
            ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(appearance["spine_width"])
        for line in ax.lines:
            line.set(**edge)
        entries = [(labels["victim_style"], styles["victim_style"])] if panel["victims"] else []
        method_legend = positive_group_colors is not None or positive_group_labels is not None
        if method_legend:
            known = {g["spec"].label for g in panel["positive_groups"]}
            require(set(positive_group_colors or {}) <= known and set(positive_group_labels or {}) <= known,
                    "Unknown positive-group label in overlay style mapping")
            start = 2*len(panel["negative_groups"])  # evaluated + reference per group
            for i, group in enumerate(panel["positive_groups"]):
                original_label = group["spec"].label
                style = {**styles["positive_style"],
                         "color": (positive_group_colors or {}).get(original_label, styles["positive_style"]["color"])}
                ax.collections[start+i].set_facecolor(style["color"])
                entries.append(((positive_group_labels or {}).get(original_label, original_label), style))
        elif panel["positive_groups"]:
            entries.append((labels["positive_style"], styles["positive_style"]))
        entries.extend((labels[role], styles[role]) for role in ("negative_style", "reference_style"))
        handles = [Line2D([], [], linestyle="none", label=label,
                          marker=style["marker"], markerfacecolor=style["color"],
                          markeredgecolor=style.get("edgecolors", "none"),
                          markeredgewidth=style.get("linewidths", 0),
                          markersize=np.sqrt(style["s"])) for label, style in entries]
        ax.legend(handles=handles, prop=FontProperties(family=font_family, size=fonts["legend"]), **legend)
        fig.canvas.draw()
        for text in fig.findobj(Text):
            text.set_fontfamily(font_family)
        fig.canvas.draw()
    except Exception:
        plt.close(fig)
        raise
    if show:
        plt.show()
    return fig, ax, summary, points


def style_reference_open_marker(ax, *, marker="+", color="#346FA7", area=16.0,
                                linewidth=.9, zorder=4.5,
                                legend_label="reference models"):
    """Style an existing single-H0 reference collection and its legend.

    The scatter order from plot_fixed_split_plane is evaluated negatives,
    references, positives, victim. Only marker appearance changes; offsets,
    null boundary, and reported decisions are left untouched.
    """
    from matplotlib.markers import MarkerStyle

    require(marker in ("o", "+", "x"), "reference marker must be o, + or x")
    require(np.isfinite([area, linewidth, zorder]).all() and area > 0 and linewidth > 0,
            "reference marker area and line width must be positive")
    require(len(ax.collections) >= 3, "Expected evaluated, reference, and victim collections")
    legend = ax.get_legend()
    require(legend is not None, "Expected a reference legend")
    reference_indices = [i for i, item in enumerate(legend.get_texts())
                         if item.get_text() == legend_label]
    require(len(reference_indices) == 1, "Reference legend must appear exactly once")

    reference = ax.collections[1]
    shape = MarkerStyle(marker)
    reference.set_paths([shape.get_path().transformed(shape.get_transform())])
    reference.set_facecolors("none")
    reference.set_edgecolors(color)
    reference.set_linewidths(linewidth)
    reference.set_sizes([area])
    reference.set_zorder(zorder)

    handle = legend.legend_handles[reference_indices[0]]
    handle.set_marker(marker)
    handle.set_markerfacecolor("none")
    handle.set_markeredgecolor(color)
    handle.set_markeredgewidth(linewidth)
    handle.set_markersize(np.sqrt(area))
    return reference


def combine_grid_row(panels, row):
    """Overlay four methods, sharing one unchanged negative/reference/victim set.

    Reject incompatible anchors instead of silently switching pools or merging
    architectures. Deep copies keep the original grid and single panels intact.
    """
    from copy import deepcopy
    require(len(panels) == 8 and row in (0, 1), "Choose row 0 or 1 of eight panels")
    selected = panels[4*row:4*row+4]
    base = deepcopy(selected[0])
    for panel in selected[1:]:
        for key in ("negative_groups", "victims", "options"):
            require(panel.get(key) == selected[0].get(key),
                    f"Cannot overlay panels with different {key}")
    base["positive_groups"] = [deepcopy(group) for panel in selected for group in panel["positive_groups"]]
    names = [group["spec"].label for group in base["positive_groups"]]
    require(len(set(names)) == len(names), "Overlay positive labels must be unique")
    return base


def save_panel_figures(figures, paths, *, dpi=600, pad_inches=.02, shared_crop=True):
    """Export panels with a common tight bounding box for consistent LaTeX sizing.

    All panels must use the same canvas size. A common crop prevents different
    tick-label widths from changing the apparent scale after LaTeX placement.
    """
    from matplotlib.transforms import Bbox
    require(len(figures) == len(paths) and len(figures) > 0, "One output path per figure is required")
    require(np.isfinite(pad_inches) and pad_inches >= 0, "Padding must be nonnegative")
    boxes = []
    for fig in figures:
        require(np.allclose(fig.get_size_inches(), figures[0].get_size_inches()),
                "Independent panels must share their canvas size")
        fig.canvas.draw()
        boxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    crop = Bbox.union(boxes).padded(pad_inches) if shared_crop else "tight"
    for fig, path in zip(figures, paths):
        save_paper_figure(fig, path, dpi=dpi, bbox_inches=crop, pad_inches=pad_inches)


def save_paper_figure(fig, path, dpi=600, *, bbox_inches=None, pad_inches=.02):
    """Vector PDF export; pass bbox_inches='tight' to crop unused outer space.

    None preserves the canvas dimensions for existing callers. Tight export
    changes the PDF page size, but does not stretch the square plotting axes.
    """
    import matplotlib.pyplot as plt
    require(np.isfinite(pad_inches) and pad_inches >= 0,
            "pad_inches must be finite and nonnegative")
    with plt.rc_context({"savefig.bbox": None, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
