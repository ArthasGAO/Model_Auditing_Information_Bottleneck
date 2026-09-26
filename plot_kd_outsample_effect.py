"""
Effect-size view of the out-sample MI signal.

Each bar is one suspect model's I(X;T)-Out offset from its ARCHITECTURE- and
RATE-matched independent-negative band. Zero = indistinguishable from an
independently trained model of the same architecture.

The ordering is the point: the magnitude tracks how much of the attacker's
supervision constrained the OUTPUT DISTRIBUTION.
    FitNet (feature MSE, 0% output supervision) ~ 0
    DKD / KD  (partial / 90% output supervision) > 0
    Knockoff  (pure soft-label KL, no CE at all) >> 0
"""
import csv
import statistics as st
import matplotlib.pyplot as plt

VAN = "./saved_logs/vanilla/MI_master_table.csv"
KD = "./saved_logs/kd_vanilla/MI_master_table_kd.csv"
EX = "./saved_logs/extraction_vanilla/MI_master_table_extraction.csv"
BINS, IN_SIZE, COL = "50", "25000", "I(X;T)-Out"


def rows(path):
    return [r for r in csv.DictReader(open(path))
            if r.get("bins") == BINS and r.get("in_size") == IN_SIZE]


V = rows(VAN)


def band(scenario, rate):
    """Mean and sd of the matched negative pool."""
    g = [float(r[COL]) for r in V if r["Scenario"] == scenario
         and r["rate"] in (rate, rate.rstrip("0").rstrip("."))]
    return st.mean(g), st.pstdev(g)


def ctrl_scenario(model_name):
    if "18to18" in model_name or "Same18" in model_name:
        return "CIFAR-10_ResNet-18_25000"
    if "VGG16" in model_name or "Cross16" in model_name:
        return "CIFAR-10_VGG16_25000"
    return "CIFAR-10_DeiT_Plain_25000"


bars = []   # (label, diff, z, colour)

COLOUR = {"FitNet": "#7f7f7f", "DKD": "#9467bd", "KD": "#d62728", "Knockoff": "#1f77b4"}

seen = set()
for r in sorted(rows(KD), key=lambda x: x["model_name"]):
    n = r["model_name"]
    if n in seen:
        continue
    seen.add(n)
    method = "FitNet" if "FitNet" in n else ("DKD" if "DKD" in n else "KD")
    cell = "A" if r["rate"] == "1.0" else "C"
    m, s = band(ctrl_scenario(n), r["rate"])
    v = float(r[COL])
    short = (n.replace("CIFAR-10_ResNet-", "").replace("_25000", "")
              .replace(f"_{method}_42_{r['rate']}", ""))
    bars.append((f"{short} {method} [cell {cell}]", v - m, (v - m) / s, COLOUR[method]))

# Knockoff reference: CIFAR-10, R18 victim, disjoint same-distribution transfer set
seen = set()
for r in sorted(rows(EX), key=lambda x: x["model_name"]):
    n = r["model_name"]
    if not n.startswith("CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18") or n in seen:
        continue
    seen.add(n)
    m, s = band("CIFAR-10_ResNet-18_25000", "0.0")
    v = float(r[COL])
    bars.append((f"Knockoff Same10_Same18 (run {n.split('_')[-2]})",
                 v - m, (v - m) / s, COLOUR["Knockoff"]))

bars.sort(key=lambda b: b[1])

fig, ax = plt.subplots(figsize=(11, 0.42 * len(bars) + 2.2))
ypos = range(len(bars))
ax.barh(list(ypos), [b[1] for b in bars],
        color=[b[3] for b in bars], alpha=0.85, height=0.68)

# +/-3 sigma of the tightest negative band, as a "indistinguishable" reference
_, s_ref = band("CIFAR-10_ResNet-18_25000", "0.0")
ax.axvspan(-3 * s_ref, 3 * s_ref, color="k", alpha=0.10, zorder=0,
           label=f"±3σ of negatives (σ={s_ref:.3f})")
ax.axvline(0, color="k", lw=1.2)

for i, (lab, d, z, _) in enumerate(bars):
    ax.text(d + (0.045 if d >= 0 else -0.045), i, f"{d:+.2f}  (z={z:+.0f})",
            va="center", ha="left" if d >= 0 else "right", fontsize=9)

ax.set_yticks(list(ypos))
ax.set_yticklabels([b[0] for b in bars], fontsize=9)
ax.set_xlabel("I(X;T)-Out  −  matched independent-negative mean   (bits)", fontsize=12)
ax.set_title("Out-sample MI offset from architecture- and rate-matched negatives\n"
             "magnitude tracks how much supervision constrained the output distribution",
             fontsize=12)
ax.margins(x=0.22)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("figures/kd_outsample_effect.png", dpi=150, bbox_inches="tight")
print("saved figures/kd_outsample_effect.png")
for lab, d, z, _ in bars:
    print(f"  {lab:<44}{d:+8.3f}  z={z:+8.1f}")
