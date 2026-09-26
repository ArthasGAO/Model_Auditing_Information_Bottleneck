%matplotlib inline
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clear_csv_cache()

CSV_KD  = "./saved_logs/kd_final/MI_master_table_kd.csv"
CSV_NEG = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_TCH = "./saved_logs/kd_final/MI_master_table_teacher.csv"
CSV_OLD = "./saved_logs/kd_vanilla/MI_master_table_kd.csv"   # ResNet-18-teacher row

BINS, IN_SIZE = 50, 25000
RATE = 0.0                   # Cell C: transfer set disjoint from group_A
POOL_SEEDS = None            # all 80 per pool
SHOW_LEGACY = True           # overlay the older ResNet-18-teacher row

# One panel per STUDENT architecture: MI is read from the student's logits, so
# the student decides which negative pool is the null. This is the same rule
# the extraction plots apply to the surrogate.
# student -> (pool scenario, KD Scenario_Name, teacher scenario, teacher label,
#             legacy ResNet-18-teacher Scenario_Name)
STUDENT = {
    "ResNet-18": ("CIFAR-10_ResNet-18_25000", "CIFAR-10_ResNet-34to18_25000",
                  "CIFAR-10_ResNet-34_25000", "ResNet-34",
                  "CIFAR-10_ResNet-18to18_25000"),
    "VGG16":     ("CIFAR-10_VGG16_25000",     "CIFAR-10_VGG19toVGG16_25000",
                  "CIFAR-10_VGG19_25000",     "VGG19",
                  "CIFAR-10_ResNet-18toVGG16_25000"),
}
# distillation method -> (colour, marker)
METHOD = {
    "KD":  ("#0072B2", "o"),
    "DKD": ("#D55E00", "s"),
}

def _sel(path):
    d = pd.read_csv(path)
    return d[(d.bins == BINS) & (d.in_size == IN_SIZE)]

_neg = _sel(CSV_NEG)
if POOL_SEEDS is not None:
    _neg = _neg[_neg.seed.isin(POOL_SEEDS)]
_kd  = _sel(CSV_KD)
_tch = _sel(CSV_TCH)
_old = _sel(CSV_OLD) if SHOW_LEGACY else None
_C = ["I(X;T)-In", "I(T;Y)-In"]

def _null(pool_scen):
    P = _neg[_neg.Scenario == pool_scen][_C].to_numpy()
    if len(P) < 3:
        raise ValueError(f"negative pool {pool_scen!r} has {len(P)} rows -- check the name")
    mu, Si = P.mean(0), np.linalg.inv(np.cov(P.T))
    r = np.sort([np.sqrt((p - mu) @ Si @ (p - mu)) for p in P])
    return mu, Si, r[int(0.95 * len(r))], len(P)

def maha(pool_scen, frame, model_re, required=True):
    """(distance, n). required=False returns (None, 0) instead of raising."""
    mu, Si, _, _ = _null(pool_scen)
    G = frame[frame.model_name.str.match(model_re, na=False)][_C].to_numpy()
    if len(G) == 0:
        if required:
            raise ValueError(f"no rows match {model_re!r} -- check the scenario / rate")
        return None, 0
    d = G.mean(0) - mu
    return float(np.sqrt(d @ Si @ d)), len(G)

p95s = [_null(s[0])[2] for s in STUDENT.values()]

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
for ax, (student, (pool_scen, kd_scen, t_scen, t_label, legacy)) in zip(axes, STUDENT.items()):
    npool = _null(pool_scen)[3]
    specs = [TableGroupSpec(
        label=f"Negative pool  (n={npool})",
        csv_path=CSV_NEG, scenario=pool_scen, rates=[0.0], seeds=POOL_SEEDS,
        bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
        style={"marker": "o", "color": "#BBBBBB", "s": 46,
               "alpha": 0.75, "edgecolors": "none", "zorder": 1},
    )]

    # the bigger same-family teacher, one group per distillation method
    for m, (colour, marker) in METHOD.items():
        rex = rf"{re.escape(kd_scen)}_{m}_\d+_{re.escape(str(RATE))}$"
        d, n = maha(pool_scen, _kd, rex)
        specs.append(TableGroupSpec(
            label=f"{t_label} teacher, {m}  (n={n}, d={d:.0f})",
            csv_path=CSV_KD, model_name=re.compile(rex),
            bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
            style={"marker": marker, "color": colour, "s": 115, "alpha": 0.95,
                   "edgecolors": "black", "linewidths": 0.6, "zorder": 3},
        ))

    # the older ResNet-18-teacher row, same student, hollow markers
    if SHOW_LEGACY:
        for m, (colour, marker) in METHOD.items():
            rex = rf"{re.escape(legacy)}_{m}_42_{re.escape(str(RATE))}$"
            d, n = maha(pool_scen, _old, rex, required=False)
            if n == 0:
                print(f"[skip] no legacy rows for {legacy} {m} at "
                      f"bins={BINS}, in_size={IN_SIZE}")
                continue
            specs.append(TableGroupSpec(
                label=f"ResNet-18 teacher, {m}  (n={n}, d={d:.0f})",
                csv_path=CSV_OLD, model_name=re.compile(rex),
                bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
                style={"marker": marker, "color": "none", "s": 130,
                       "edgecolors": colour, "linewidths": 1.6, "zorder": 2},
            ))

    dt, _ = maha(pool_scen, _tch, rf"{re.escape(t_scen)}_42_1\.0$")
    teacher = VictimSpec(
        csv_path=CSV_TCH, scenario=t_scen,
        seeds=[42], rates=[1.0], bins=BINS, in_size=IN_SIZE, domain="in",
        label=f"{t_label} teacher  (d={dt:.0f})",
        style={"marker": "*", "color": "#CC79A7", "s": 320,
               "edgecolors": "black", "linewidths": 0.8, "zorder": 4},
    )

    plot_information_plane(
        specs, title=f"student = {student}      null = {student} pool",
        victim=[teacher], ax=ax, show=False, legend_outside=False,
        font_sizes={"title": 13, "xlabel": 13, "ylabel": 13,
                    "xtick": 11, "ytick": 11, "legend": 9},
    )

fig.suptitle(
    f"Knowledge distillation, CIFAR-10 / Cell C (transfer set disjoint from group_A)\n"
    f"d = Mahalanobis distance of the group mean from its own null "
    f"(pool p95 radius {min(p95s):.1f}-{max(p95s):.1f});  bins={BINS}, in_size={IN_SIZE}",
    fontsize=14, y=1.05)
fig.tight_layout()
plt.show()
