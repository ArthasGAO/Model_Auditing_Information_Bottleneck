# Implementation Guide: Generalizing the First-Order Boundary-Distance Estimator to $L_p$

**Scope.** Extend the current $L_2$ closed-form estimator in `boundary_band.py` to a general $\ell_p$ metric. This guide lists what changes, what does *not*, the paper-traceable formulas, and the numerical/correctness caveats surfaced in our analysis.

---

## 0. Core principle (applies to every module below)

> **Numerator = output/margin side; invariant in $p$.**
> **Denominator = input/geometry side; $p$ acts here via the dual norm $q$, with $\tfrac{1}{p}+\tfrac{1}{q}=1$.**

Anything touching distance / perturbation / geometry must be swept over $p$. Anything touching only margins, logits, or class selection stays as-is.

**Convention.** The user-facing parameter is $p$ (the distance metric). Convert to $q$ *internally* before calling any norm — never pass $q$ from the caller (avoids mental dual-norm arithmetic and off-by-one errors).

| $p$ | $q=\tfrac{p}{p-1}$ | Status |
|---|---|---|
| $2$ | $2$ | Self-dual. Regression baseline. |
| $\infty$ | $1$ | Paper eq. (13–14). Primary sweep target. |
| $1$ | $\infty$ | **Outside DeepFool's $[1,\infty)$ framework**; single-coordinate degeneracy. Optional/experimental only. |

---

## 1. Distance closed form — **CHANGE**

Replace the fixed $L_2$ denominator with the dual $q$-norm:

$$
\delta^{(1)}_{j,p}(x_0) \;=\; \frac{|g_{k,j}(x_0)|}{\lVert \nabla g_{k,j}(x_0)\rVert_q}, \qquad q=\tfrac{p}{p-1}.
$$

- Code: `margin / grad.norm(2)` → `margin / grad.norm(q)`.
- Handle `q = float('inf')` (i.e. $p=1$) explicitly: `torch.norm(x, float('inf'))` = $\max_i|x_i|$.
- Numerator `|g_{k,j}|` unchanged. (For predicted class $k$, $g_{k,j}\ge 0$, so `abs` is a no-op but keep it for the iterative/general path.)

---

## 2. Perturbation direction $r$ — **CHANGE (only if $r$ is used)**

Skip this section if the estimator returns the scalar $\delta$ only. Required if $r$ is used anywhere (post-hoc check, export, visualization).

**General form** (DeepFool eq. 12, our single-shot sign convention with $b=g_{k,j}\ge 0$):

$$
r \;=\; -\,b\;\frac{\operatorname{sign}(a)\odot |a|^{q-1}}{\lVert a\rVert_q^{\,q}}, \qquad a=\nabla g_{k,j}(x_0),\ \ b=g_{k,j}(x_0).
$$

**Sign note.** DeepFool writes $+|f'|$ (eq. 12) because in the iterative loop the running margin $f'$ carries orientation; our single-shot form uses $-b$ to *decrease* the margin from the predicted-class side. Both land on the same side of the boundary — the difference is iterative bookkeeping, not a discrepancy.

**Mandatory sanity checks (write as unit tests):**
- $q=2$: reduces to $-b\,a/\lVert a\rVert_2^2$ → must be element-wise equal to the old code. **This is the regression test.**
- $q=1$ ($p=\infty$, DeepFool eq. 14): $|a|^{q-1}=|a|^0=1$ ⇒ $r=-b\,\operatorname{sign}(a)/\lVert a\rVert_1$ (FGSM direction).
- $q=\infty$ ($p=1$): $|a|^{q-1}$ is a singular limit — **do not use the general formula**. Branch: put mass on $\arg\max_i|a_i|$, zero elsewhere. Numerically unstable when the top-$2$ coordinates are near-tied; expect long whiskers. Outside the cited framework.

---

## 3. Post-hoc dominance verification — **NO FORMULA CHANGE**

The check $g_{k,\ell}(x_0+r)\ge 0,\ \forall \ell\notin\{k,j\}$ is a pure output-side class-membership test → criterion unchanged for all $p$. Only dependency: it must consume the $p$-correct $r$ from §2 (guard against a stale $L_2$ residual $r$). Add a unit test asserting pass-rate at $q=2$ equals the old version.

*(See §8 for the one open design question about whether this criterion should be $p$-coupled.)*

---

## 4. Multi-class aggregation $\min_j$ / $\arg\min_j$ — **NO CODE CHANGE, semantic caveat**

$$
\Delta^{(1)}_p = \min_{j\neq k}\delta^{(1)}_{j,p}, \qquad j^\star_p = \arg\min_{j\neq k}\delta^{(1)}_{j,p}.
$$

Confirmed by DeepFool eq. (11): the selection uses the $q$-norm denominator.

**Caveat:** $j^\star$ can differ across $p$ — different $p$ may report distance to a *different facet*, not two measurements of the same one. When comparing across $p$, log $j^\star_p$ per point and report the fraction where $j^\star$ agrees, so facet-switching doesn't masquerade as a metric effect.

---

## 5. Top-$K$ competitor selection — **NO CODE CHANGE, reliability drops**

Top-$K$ ranks by margin $g_j$ (output side) → code is $p$-independent. But its validity rests on "small margin ≈ near class," which is already only approximate at $L_2$ and **loosens further as $q\to 1$**: distance is $g_j/\lVert\nabla g_j\rVert_q$, yet top-$K$ ignores the denominator entirely. Since $\lVert\cdot\rVert_1$ can exceed $\lVert\cdot\rVert_2$ by up to $\sqrt{D}$ ($\approx 55$ for CIFAR), the denominator's weight in the true ranking grows, so top-$K$ is likelier to miss the true nearest class under $\ell_\infty$.

**Action (diagnostic, not a code change):** on a small subset, run full-$C$ vs top-$K$ and report $\arg\min$ agreement per $p$. If $\ell_\infty$ agreement drops, raise $K$ or compute full-$C$ for $\ell_\infty$.

---

## 6. Threshold calibration `calibrate_d` — **NO CODE CHANGE, interpretation discipline**

Criterion (1st-percentile of test $\delta$) unchanged; feed the $p$-version array to get $d_p$ automatically.

> **Cross-$p$ absolute values are NOT comparable.** $\ell_\infty$ distances are $1$–$2$ orders of magnitude smaller than $\ell_2$ purely because $\lVert a\rVert_1\ge\lVert a\rVert_2$ — a unit shift, not a separability change. Box plots placed side-by-side across $p$ will mislead.

**Compare only dimensionless separability:** AUROC (test = neg, fp = pos, $\delta$ = score), fp-median / test-median ratio, or recall@FPR=1%. Emit these as standard output right after calibration.

---

## 7. Numerical stability — **CHANGE, $q$-dependent**

- **Denominator underflow guard** (§1): $\ell_\infty$ ($q=1$) triggers differently than $L_2$ — re-tune the threshold per $q$, don't reuse the $L_2$ constant.
- **$|a|^{q-1}$ overflow/underflow** (§2): large $q$ can over/underflow; $\ell_1$ ($q=\infty$) must take the single-coordinate branch, never the general expression.
- **Dead-code guard** (the known `delta2 < 0` remnant): clean up or fix while touching this path so it doesn't re-activate on the new branch.

---

## Change / no-change summary

| Module | Formula change? | Action |
|---|---|---|
| §1 Distance | **Yes** | `.norm(2)→.norm(q)`; caller passes $p$, convert to $q$; special-case $q=1,\infty$ |
| §2 Direction $r$ | **Yes** (if used) | $\operatorname{sign}(a)\,|a|^{q-1}$ form; $p=1$ single-coord branch; $q=2$ regression |
| §3 Post-hoc | No | Criterion fixed; feed new $r$; test pass-rate at $q=2$ |
| §4 min/argmin | No | Structure fixed; log $j^\star_p$, report agreement |
| §5 Top-$K$ | No (code) | Add full-$C$ vs top-$K$ agreement diagnostic per $p$ |
| §6 calibrate_d | No | Auto-updates; **add dimensionless metrics; ban cross-$p$ absolute comparison** |
| §7 Numerics | **Yes** | Re-tune guards per $q$; clean dead code |

---

## Recommended implementation order (each step independently verifiable)

1. **§1 scalar only.** Add $p$; run $q=2$; assert element-wise equality with old output (regression). Lowest risk — no direction, no verification touched.
2. **§2 direction.** Align to old $r$ at $q=2$; verify FGSM direction at $p=\infty$.
3. **§3 post-hoc.** Confirm pass-rate matches old at $q=2$.
4. **§6 + §5.** Add dimensionless separability metrics and the full-$C$ vs top-$K$ diagnostic; run the sweep.

**Sweep scope.** Primary: $p\in\{2,\infty\}$ — both have closed forms traceable to DeepFool eq. (12) / (13–14) and are citable directly. Optional/experimental: $p=1$, with the explicit single-coordinate branch, **not** justified by the cited paper.

---

## 8. Open design decision (confirm before coding §3)

Should post-hoc dominance verification stay $p$-independent, or become $p$-coupled?

- **Default (recommended):** keep it $p$-independent. Dominance is a pure class-membership judgment (is a third class winning at the candidate point?), conceptually orthogonal to which $\ell_p$ you measure distance in.
- **Alternative:** if your band semantics require a stronger joint criterion ("inside the band in the $\ell_p$ sense"), the check must be redefined. This depends on the band definition in your report — confirm which one you intend, as it decides whether §3 is a zero-change or a redesign.

---

*Formula provenance: distance and direction — DeepFool $\ell_p$ update eqs. (11)–(14) and the Large-Margin closed form $\tilde d = |f_i-f_j|/\lVert\nabla f_i-\nabla f_j\rVert_q$; caveats (unit-shift, top-$K$ looseness, facet-switching, relaxation-vs-linearization) — from this conversation's analysis. Verify the paper's footnote 3 on $q=\tfrac{p}{p-1}$ for any tightness qualification before finalizing the writeup.*
