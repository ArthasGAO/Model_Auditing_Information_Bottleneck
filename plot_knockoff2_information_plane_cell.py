%matplotlib inline
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clear_csv_cache()

CSV_EXT  = "./saved_logs/extraction_final/MI_master_table_extraction.csv"
# The 1-hop CIFAR-100-queried surrogates were trained before extraction_final/
# existed and still live in extraction_vanilla/. Their stored MI is on the old
# 10-row grid, so it was recomputed on the shared 70-cell grid into this
# separate file rather than appended to the single-writer table above.
CSV_OOD1 = "./saved_logs/extraction_vanilla/MI_master_table_extraction_grid70.csv"
CSV_NEG  = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_VIC  = "./saved_logs/vanilla/MI_master_table_victim.csv"

BINS, IN_SIZE = 50, 25000
VICTIM = "CIFAR-10_ResNet-18_25000"

# One panel per chain, and the panel's null is the SUSPECT's pool, because MI is
# read from the suspect's logits. Each chain's own intermediate belongs to a
# different pool than its hop-2 model, so no intermediate is drawn: chain B runs
# through a VGG16 and chain C through a DeiT, and neither shares an axis with
# the model it produced.
#
# colour = hop-2 query set, fill = hop count. Reading a Cross100 point against
# a Same10 point is meaningless: the query set alone moves d by 2-5x, almost
# all of it along I(X;T). Compare within a colour.
PANEL = [
    # (panel title, pool scenario, [(hop-2 query set, csv, model_name regex, hops)])
    ("chain A   RN18 -> RN18 -> RN18", "CIFAR-10_ResNet-18_25000", [
        ("Same10",   CSV_EXT,  rf"{VICTIM}_Knockoff_Same10_Same18_\d+_1\.0$",              1),
        ("Same10",   CSV_EXT,  rf"{VICTIM}_Knockoff2_Same10_Via18_Same10_Same18_\d+_1\.0$", 2),
        ("Cross100", CSV_OOD1, rf"{VICTIM}_Knockoff_Cross100_Same18_\d+_1\.0$",            1),
        ("Cross100", CSV_EXT,  rf"{VICTIM}_Knockoff2_Same10_Via18_Cross100_Same18_\d+_1\.0$", 2),
    ]),
    ("chain B   RN18 -> VGG16 -> DeiT-Ti", "CIFAR-10_DeiT_Plain_25000", [
        ("Same10",   CSV_EXT,  rf"{VICTIM}_Knockoff_Same10_CrossDeiT_\d+_1\.0$",              1),
        ("Same10",   CSV_EXT,  rf"{VICTIM}_Knockoff2_Same10_Via16_Same10_CrossDeiT_\d+_1\.0$", 2),
        ("Cross100", CSV_OOD1, rf"{VICTIM}_Knockoff_Cross100_CrossDeiT_\d+_1\.0$",            1),
        ("Cross100", CSV_EXT,  rf"{VICTIM}_Knockoff2_Same10_Via16_Cross100_CrossDeiT_\d+_1\.0$", 2),
    ]),
    # Chain C is chain B reversed: out to the transformer and back to a CNN.
    # Its hop-2 models are VGG16, so it is read against the VGG16 pool and its
    # one-hop references are the VGG16 surrogates of the same victim.
    ("chain C   RN18 -> DeiT-Ti -> VGG16", "CIFAR-10_VGG16_25000", [
        ("Same10",   CSV_EXT,  rf"{VICTIM}_Knockoff_Same10_Cross16_\d+_1\.0$",                1),
        ("Same10",   CSV_EXT,  rf"{VICTIM}_Knockoff2_Same10_ViaDeiT_Same10_Cross16_\d+_1\.0$", 2),
        ("Cross100", CSV_OOD1, rf"{VICTIM}_Knockoff_Cross100_Cross16_\d+_1\.0$",              1),
        ("Cross100", CSV_EXT,  rf"{VICTIM}_Knockoff2_Same10_ViaDeiT_Cross100_Cross16_\d+_1\.0$", 2),
    ]),
]
QUERY = {                       # hop-2 query set -> colour
    "Same10":   "#0072B2",
    "Cross100": "#D55E00",
}
HOPS = {                        # hop count -> marker, and hollow vs filled
    1: ("o", False),
    2: ("s", True),
}
_C = ["I(X;T)-In", "I(T;Y)-In"]


def _sel(path):
    d = pd.read_csv(path)
    return d[(d.bins == BINS) & (d.in_size == IN_SIZE)]


_frames = {p: _sel(p) for p in (CSV_EXT, CSV_OOD1, CSV_NEG, CSV_VIC)}
_neg = _frames[CSV_NEG]


def _null(pool_scen):
    P = _neg[_neg.Scenario == pool_scen][_C].to_numpy()
    if len(P) < 3:
        raise ValueError(f"negative pool {pool_scen!r} has {len(P)} rows -- check the name")
    mu, Si = P.mean(0), np.linalg.inv(np.cov(P.T))
    r = np.sort([np.sqrt((p - mu) @ Si @ (p - mu)) for p in P])
    return mu, Si, r[int(0.95 * len(r))], len(P)


def maha(pool_scen, frame, model_re):
    """(d in units of the pool p95, n). Raises if the regex matches nothing."""
    mu, Si, p95, _ = _null(pool_scen)
    G = frame[frame.model_name.str.match(model_re, na=False)][_C].to_numpy()
    if len(G) == 0:
        raise ValueError(f"no rows match {model_re!r} -- check the scenario / rate")
    v = G.mean(0) - mu
    return float(np.sqrt(v @ Si @ v)) / p95, len(G)


fig, axes = plt.subplots(1, 3, figsize=(21.5, 5.8))
p95s = []
for ax, (title, pool_scen, cells) in zip(axes, PANEL):
    mu, _, p95, npool = _null(pool_scen)
    p95s.append(p95)

    specs = [TableGroupSpec(
        label=f"Negative pool  (n={npool})",
        csv_path=CSV_NEG, scenario=pool_scen, rates=[0.0],
        bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
        style={"marker": "o", "color": "#BBBBBB", "s": 46,
               "alpha": 0.75, "edgecolors": "none", "zorder": 1},
    )]

    for query, csv_path, rex, hops in cells:
        colour = QUERY[query]
        marker, filled = HOPS[hops]
        d, n = maha(pool_scen, _frames[csv_path], rex)
        style = {"marker": marker, "s": 125, "zorder": 2 + hops,
                 "linewidths": 1.7, "edgecolors": colour, "color": "none"}
        if filled:
            style.update({"color": colour, "edgecolors": "black",
                          "linewidths": 0.6, "alpha": 0.95})
        specs.append(TableGroupSpec(
            label=f"{hops} hop{'s' if hops > 1 else ' '}, {query:<8} (n={n}, d={d:.1f})",
            csv_path=csv_path, model_name=re.compile(rex),
            bins=BINS, in_size=IN_SIZE, domain="in", mode="points", style=style,
        ))

    # The victim is a ResNet-18, so it can only be drawn against the ResNet-18
    # null. On the DeiT panel it would be a point from a different logit
    # geometry plotted on the wrong axes.
    victim = []
    if pool_scen == VICTIM:
        dv, _ = maha(pool_scen, _frames[CSV_VIC], rf"{VICTIM}_42_1\.0$")
        victim = [VictimSpec(
            csv_path=CSV_VIC, scenario=VICTIM, seeds=[42], rates=[1.0],
            bins=BINS, in_size=IN_SIZE, domain="in",
            label=f"victim  (d={dv:.1f})",
            style={"marker": "*", "color": "#CC79A7", "s": 320,
                   "edgecolors": "black", "linewidths": 0.8, "zorder": 5},
        )]

    plot_information_plane(
        specs, title=f"{title}      null = {pool_scen.split('_')[1]} pool",  # noqa: E501
        victim=victim, ax=ax, show=False, legend_outside=False,
        font_sizes={"title": 13, "xlabel": 13, "ylabel": 13,
                    "xtick": 11, "ytick": 11, "legend": 9},
    )

fig.suptitle(
    "Double knockoff extraction, CIFAR-10 / victim = ResNet-18\n"
    "hollow = 1 hop (victim -> S), filled = 2 hops (victim -> S1 -> S2);  "
    "colour = hop-2 query set;  each panel's null is its own suspect's pool\n"
    f"d = Mahalanobis distance of the group mean from its own null, in units of "
    f"that pool's p95 ({min(p95s):.1f}-{max(p95s):.1f} raw);  bins={BINS}, in_size={IN_SIZE}",
    fontsize=13, y=1.10)
fig.tight_layout()
plt.show()
