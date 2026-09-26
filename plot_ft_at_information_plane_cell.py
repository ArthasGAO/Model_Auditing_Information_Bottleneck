# %% [markdown]
# Information plane: FT-AL with an added adversarial term (ft_at) against its
# references. Self-contained (pandas / numpy / matplotlib / scipy only), so it
# runs in any notebook without the TableGroupSpec helpers. Copy the whole file
# into one cell, or run it as a script (it then saves PNGs next to the CSVs).
#
# Groups drawn at one (in_size, bins) operating point, In-domain MI:
#   * negative pool: independent ResNet-18 models, rate 0.0 (the H0 reference);
#     grey points + its 95% Mahalanobis ellipse;
#   * victim (star);
#   * plain FT-AL (ft_final, 3 seeds): the attack WITHOUT the adversarial term;
#   * ft_at FT-AL + lambda * CE(x_adv) at eps 2/255, 4/255, 8/255 (3 seeds each);
#   * post-hoc AT of the victim (at_final: main_at.py, mix=off, lr 0.01, 30 ep,
#     eps 2/4/8, 5 seeds, best_clean checkpoint) as hollow diamonds, i.e. the
#     two-stage baseline the one-stage variant is compared with;
#   * AT applied to INDEPENDENT rate-0.0 models (at_vanilla, PGD, best_clean;
#     both the from-scratch `atseed` and the fine-tuned `ftseed` variants) as
#     hollow triangles: the false-positive control. If these sit as far from
#     the pool as the derived models, the pool of non-robust models cannot
#     separate "derived and robust" from "independent and robust", and only
#     the I(T;Y) axis carries the derivation signal.
# The right panel shows the Mahalanobis distance to the pool mean (using the
# pool covariance) as a function of eps; the third figure sweeps the whole
# (in_size, bins) grid so the conclusion is not tied to one operating point.

# %%
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

CSV_NEG = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_VIC = "./saved_logs/vanilla/MI_master_table_victim.csv"
CSV_FT = "./saved_logs/ft_final/MI_master_table_ft.csv"
CSV_FTAT = "./saved_logs/ft_at/MI_master_table_ft_at.csv"
CSV_AT = "./saved_logs/at_final/MI_master_table_at.csv"
CSV_ATN = "./saved_logs/at_vanilla/MI_master_table_at_neg.csv"   # AT on independent models (in_size 25000 only)

POOL_SCEN = "CIFAR-10_ResNet-18_25000"            # pool scenario == victim scenario
FT_SCEN = "CIFAR-10_ResNet-18_25000_Same_25000"   # main_ft / main_ft_at Scenario_Name
VICTIM_AT_BASE = "CIFAR-10_ResNet-18_25000_42_1.0"  # at_final base_model of the victim
IN_SIZE, BINS = 25000, 50
DOMAIN = "In"                                     # "In" = probing set D_V; "Out" = test set
ALPHA = 0.05
SAVE_PNG = None                                   # e.g. "./figures/ft_at_information_plane.png"

EPS = {0.007843: ("2/255", "#E69F00"), 0.015686: ("4/255", "#D55E00"), 0.031373: ("8/255", "#B2182B")}
_C = [f"I(X;T)-{DOMAIN}", f"I(T;Y)-{DOMAIN}"]
FT_AL_RE = rf"{re.escape(FT_SCEN)}_42_1\.0_FT-AL_ftsize=25000_ftseed=\d+$"


def load_all():
    neg = pd.read_csv(CSV_NEG)
    neg = neg[(neg.Scenario == POOL_SCEN) & (neg.rate == 0.0)]
    vic = pd.read_csv(CSV_VIC)
    vic = vic[(vic.Scenario == POOL_SCEN) & (vic.rate == 1.0)]
    ft = pd.read_csv(CSV_FT)
    ft = ft[ft.model_name.str.match(FT_AL_RE, na=False)]
    ftat = pd.read_csv(CSV_FTAT)
    ftat = ftat[ftat.strategy == "FT-AL"]
    at = pd.read_csv(CSV_AT)
    at = at[(at.base_model == VICTIM_AT_BASE) & (at.ckpt_kind == "best_clean")]
    atn = pd.read_csv(CSV_ATN)
    atn = atn[(atn.Scenario == POOL_SCEN) & (atn.rate == 0.0) & (atn.epoch == "best_clean_epoch")
              & atn.model_name.str.contains("_PGD_eps=", na=False)]
    atn = atn.assign(eps=atn.model_name.str.extract(r"_PGD_eps=([0-9.]+)_")[0].astype(float))
    return neg, vic, ft, ftat, at, atn


def at_cell(df, in_size, bins):
    return df[(df.bins == bins) & (df.in_size == in_size)]


class Null:
    """Pool mean / covariance at one operating point and Mahalanobis distances to it."""
    def __init__(self, pool_rows):
        P = pool_rows[_C].to_numpy(dtype=float)
        if len(P) < 3:
            raise ValueError(f"pool has {len(P)} rows at this operating point")
        self.P, self.mu = P, P.mean(0)
        self.S = np.cov(P.T)
        self.Si = np.linalg.inv(self.S)
        self.n = len(P)
        self.pool_d = self.d(P)
        self.d95 = float(np.quantile(self.pool_d, 0.95))
        self.d_chi2 = float(np.sqrt(chi2.ppf(1 - ALPHA, df=2)))   # asymptotic chi2(2) cut

    def d(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float)) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", X, self.Si, X))

    def ellipse(self, ax, q=1 - ALPHA, **kw):
        w, v = np.linalg.eigh(self.S)
        k = np.sqrt(chi2.ppf(q, df=2))
        angle = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        ax.add_patch(Ellipse(self.mu, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=angle, **kw))


def groups_at(neg, vic, ft, ftat, at, atn, in_size, bins):
    """[(label, points, style, kind)] at one operating point; kind in {victim, ft, ftat, at, atn}."""
    out = []
    out.append(("victim (not fine-tuned)", at_cell(vic, in_size, bins)[_C].to_numpy(),
                dict(marker="*", color="#CC79A7", s=320, edgecolors="black", linewidths=0.8, zorder=5), "victim"))
    out.append(("plain FT-AL (no adversarial term)", at_cell(ft, in_size, bins)[_C].to_numpy(),
                dict(marker="s", color="#009E73", s=110, edgecolors="black", linewidths=0.6, zorder=4), "ft"))
    for eps, (name, colour) in EPS.items():
        rows = at_cell(ftat[np.isclose(ftat.eps, eps)], in_size, bins)
        out.append((f"FT-AL + CE(x_adv), eps {name}", rows[_C].to_numpy(),
                    dict(marker="o", color=colour, s=120, edgecolors="black", linewidths=0.6, zorder=4), "ftat"))
    for eps, (name, colour) in EPS.items():
        rows = at_cell(at[np.isclose(at.eps, eps)], in_size, bins)
        out.append((f"post-hoc AT of victim, eps {name} (mix=off)", rows[_C].to_numpy(),
                    dict(marker="D", facecolors="none", edgecolors=colour, s=95, linewidths=1.6, zorder=3), "at"))
    for eps, (name, colour) in EPS.items():
        rows = at_cell(atn[np.isclose(atn.eps, eps)], in_size, bins)
        out.append((f"AT on INDEPENDENT rate-0.0 model, eps {name} (control)", rows[_C].to_numpy(),
                    dict(marker="^", facecolors="none", edgecolors=colour, s=120, linewidths=1.6, zorder=3), "atn"))
    return out


neg, vic, ft, ftat, at, atn = load_all()
null = Null(at_cell(neg, IN_SIZE, BINS))
groups = groups_at(neg, vic, ft, ftat, at, atn, IN_SIZE, BINS)

# ---- table: position of every group relative to the pool ----------------------
print(f"Operating point in_size={IN_SIZE}, bins={BINS}, {DOMAIN}-domain MI. Pool n={null.n}, "
      f"pool 95% distance={null.d95:.2f}, chi2(2) {int((1-ALPHA)*100)}% cut={null.d_chi2:.2f}")
print(f"{'group':<46} {'n':>3} {'I(X;T)':>8} {'I(T;Y)':>8} {'d mean':>7} {'d min':>6} {'d max':>6} {'reject':>7}")
print(f"{'negative pool (self)':<46} {null.n:>3} {null.mu[0]:>8.3f} {null.mu[1]:>8.3f} "
      f"{null.pool_d.mean():>7.2f} {null.pool_d.min():>6.2f} {null.pool_d.max():>6.2f} "
      f"{(null.pool_d > null.d_chi2).mean():>7.0%}")
for label, pts, _, _ in groups:
    if len(pts) == 0:
        print(f"{label:<46} {'0':>3}   (no rows at this operating point)")
        continue
    d = null.d(pts)
    print(f"{label:<46} {len(pts):>3} {pts[:, 0].mean():>8.3f} {pts[:, 1].mean():>8.3f} "
          f"{d.mean():>7.2f} {d.min():>6.2f} {d.max():>6.2f} {(d > null.d_chi2).mean():>7.0%}")

# ---- figure 1: information plane + distance vs eps -----------------------------
fig, (ax, axd) = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1.35, 1]})
ax.scatter(null.P[:, 0], null.P[:, 1], marker="o", color="#BBBBBB", s=46, alpha=0.75,
           edgecolors="none", zorder=1, label=f"negative pool, rate 0.0 (n={null.n})")
null.ellipse(ax, fill=False, edgecolor="#888888", linestyle="--", linewidth=1.2, zorder=1)
for label, pts, style, _ in groups:
    if len(pts) == 0:
        continue
    d = null.d(pts)
    ax.scatter(pts[:, 0], pts[:, 1], label=f"{label}  (n={len(pts)}, d={d.mean():.1f})", **style)
ax.set_xlabel(f"I(X;T) {DOMAIN} [bits]")
ax.set_ylabel(f"I(T;Y) {DOMAIN} [bits]")
ax.set_title(f"CIFAR-10 ResNet-18 victim, in_size={IN_SIZE}, bins={BINS}; dashed = pool 95% ellipse")
ax.legend(fontsize=8.5, loc="best")
ax.grid(alpha=0.25)

eps_x = [0.0] + list(EPS)
one = {"ftat": ("FT-AL + CE(x_adv), one-stage", "-", "o", "#B2182B"),
       "at": ("post-hoc AT of victim, two-stage", "--", "D", "#4477AA"),
       "atn": ("AT on independent rate-0.0 model (control)", ":", "^", "#555555")}
for kind, (lab, ls, mk, col) in one.items():
    xs, means, lo, hi = [], [], [], []
    for label, pts, _, k in groups:
        if k == "ft":
            base = null.d(pts)
        if k == kind and len(pts):
            m = re.search(r"eps (\d+)/255", label)
            xs.append(int(m.group(1)) / 255)
            d = null.d(pts)
            means.append(d.mean()); lo.append(d.min()); hi.append(d.max())
    if xs:
        axd.errorbar(xs, means, yerr=[np.subtract(means, lo), np.subtract(hi, means)], fmt=mk + ls,
                     color=col, capsize=4, label=f"{lab} (mean, min-max over seeds)")
ftd = null.d([p for l, p, _, k in groups if k == "ft"][0])
axd.errorbar([0.0], [ftd.mean()], yerr=[[ftd.mean() - ftd.min()], [ftd.max() - ftd.mean()]], fmt="s",
             color="#009E73", capsize=4, label="plain FT-AL (eps = 0)")
vd = null.d([p for l, p, _, k in groups if k == "victim"][0])
axd.axhline(vd[0], color="#CC79A7", linestyle=":", label=f"victim (d={vd[0]:.1f})")
axd.axhline(null.d95, color="#888888", linestyle="--", label=f"pool 95% distance ({null.d95:.2f})")
axd.axhline(null.d_chi2, color="black", linestyle=":", linewidth=1, label=f"chi2(2) cut at alpha={ALPHA} ({null.d_chi2:.2f})")
axd.set_xticks(eps_x); axd.set_xticklabels(["0", "2/255", "4/255", "8/255"])
axd.set_xlabel("PGD eps used in training")
axd.set_ylabel("Mahalanobis distance to pool mean")
axd.set_yscale("log")
axd.set_title("Distance from the H0 pool vs eps (log scale)")
axd.legend(fontsize=8.5, loc="best")
axd.grid(alpha=0.25, which="both")
fig.tight_layout()
if SAVE_PNG:
    fig.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
plt.show()

# ---- figure 2: mean distance over the whole (in_size, bins) grid ----------------
sizes = sorted(ftat.in_size.unique())
bins_all = sorted(ftat.bins.unique())
panels = [("plain FT-AL", "ft", None)] + [(f"FT-AL + CE(x_adv) eps {n}", "ftat", e) for e, (n, _) in EPS.items()]
fig2, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6), sharey=True)
vmax = 0
grids = []
for title, kind, eps in panels:
    G = np.full((len(bins_all), len(sizes)), np.nan)
    for i, b in enumerate(bins_all):
        for j, s in enumerate(sizes):
            pool_rows = at_cell(neg, s, b)
            if len(pool_rows) < 3:
                continue
            nl = Null(pool_rows)
            src = ft if kind == "ft" else ftat[np.isclose(ftat.eps, eps)]
            pts = at_cell(src, s, b)[_C].to_numpy()
            if len(pts):
                G[i, j] = nl.d(pts).mean() / nl.d_chi2     # >1 means rejected at alpha on average
    grids.append((title, G))
    vmax = max(vmax, np.nanmax(G))
for axg, (title, G) in zip(axes, grids):
    im = axg.imshow(G, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    axg.set_xticks(range(len(sizes))); axg.set_xticklabels(sizes, rotation=45, fontsize=8)
    axg.set_yticks(range(len(bins_all))); axg.set_yticklabels(bins_all, fontsize=8)
    axg.set_xlabel("in_size"); axg.set_title(title, fontsize=10)
    for i in range(len(bins_all)):
        for j in range(len(sizes)):
            if np.isfinite(G[i, j]):
                axg.text(j, i, f"{G[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                         color="white" if G[i, j] < 0.6 * vmax else "black")
axes[0].set_ylabel("bins")
fig2.colorbar(im, ax=axes, shrink=0.85, label=f"mean distance / chi2(2) cut at alpha={ALPHA}  (>1 = rejected)")
fig2.suptitle("Mean Mahalanobis distance to the rate-0.0 pool over the (in_size, bins) grid", y=1.02)
if SAVE_PNG:
    fig2.savefig(SAVE_PNG.replace(".png", "_grid.png"), dpi=150, bbox_inches="tight")
plt.show()
