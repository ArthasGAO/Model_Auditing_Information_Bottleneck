%matplotlib inline
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================================================
# Post-hoc AT trajectories in the DeepJudge Fig.2 idiom: an open corner of
# three tinted panes rather than a wireframe box.
#
#   floor  (tinted)  = the information plane on group_A   x = I(X;T), y = I(T;Y)
#   height           = the baseline's own metric          z = IPGuard TR, RobD, ...
#   colour           = epoch, 0 -> 29
#
# In plotly each axis tints the wall it is drawn against: xaxis -> the left
# pane, yaxis -> the back pane, zaxis -> the floor. That is the one mapping
# that makes this layout reproducible, and it is not in the docs by that name.
# =====================================================================
CSV_MI = "./saved_logs/at_evasion/MI_master_table_at_traj.csv"
CSV_BASE = "./saved_logs/at_eval/posthoc_c10_traj_4baselines/by_baseline/{}.csv"
CSV_PRE = {                       # where each source's own pre-AT MI row lives
    "FT-AL": ("./saved_logs/ft_final/MI_master_table_ft.csv",
              "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_FT-AL_ftsize=25000_ftseed=0"),
    "Pruning": ("./saved_logs/pruning_final/MI_master_table_prune.csv",
                "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_sparsity=0.2_FT-AL_ftsize=25000_ftseed=0_ckpt=best"),
    "DKD": ("./saved_logs/kd_final/MI_master_table_kd.csv",
            "CIFAR-10_ResNet-18to18_25000_DKD_0_0.0"),
    "Knockoff": ("./saved_logs/extraction_final/MI_master_table_extraction.csv",
                 "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18_0_1.0"),
}

FAMILY = "FT-AL"          # FT-AL | Pruning | DKD | Knockoff
BASELINE = "IPGuard"      # IPGuard | DeepJudge | ADV_TRA
METRIC = "TR"             # IPGuard: TR/TL/RR/RL/Mean · DeepJudge: RobD/JSD · ADV_TRA: Detection_Rate
EPS = 0.031373            # one trajectory per figure: 0.007843 (2/255) | 0.015686 (4/255) | 0.031373 (8/255)
                          # None -> all three eps on one figure
# tau: median over the 50 frozen rounds of the k=30 reference split -- the split
# the MI hypothesis test uses. Per-round taus span 0.38-0.42 (IPGuard TR),
# 0.4546-0.4702 (RobD), 0.0374-0.0392 (JSD), 0.06-0.08 (ADV-TRA), so the plane
# is a median, not a hard line.
TAU = {("IPGuard", "TR"): 0.4100, ("DeepJudge", "RobD"): 0.4612,
       ("DeepJudge", "JSD"): 0.0386, ("ADV_TRA", "Detection_Rate"): 0.0700}
BINS, IN_SIZE, RUN_TAG = 50, 25000, "traj2"
Z_RANGE = None            # e.g. (0, 0.15) to expand a trajectory flattened by a far-off pre-AT point
# Pin the viewing angle. Leave None for the default; to choose one, set
# INTERACTIVE = True, drag the figure until it looks right, then run
#     print(fig.layout.scene.camera)
# in the next cell and paste the dict here. A FigureWidget writes the camera
# back into the layout as you drag; a plain Figure does NOT, which is why the
# angle cannot be recovered from the HTML export.
CAMERA = None             # e.g. dict(eye=dict(x=1.65, y=-1.75, z=0.75))
INTERACTIVE = True        # True -> a live readout of the camera appears under the figure;
                          # drag, copy the line it shows into CAMERA, then set False.
# The readout is plain plotly.js: it listens to the figure's relayout events and
# prints scene.camera. FigureWidget would give the same thing on the Python side
# (fig.layout.scene.camera) but needs the anywidget package, which this
# environment does not have; this needs nothing and also works in the HTML
# export. {plot_id} is substituted by plotly with the figure's div id.
CAPTURE_CAMERA_JS = """
var gd = document.getElementById('{plot_id}');
var out = document.createElement('pre');
out.style.cssText = 'font:13px/1.5 monospace;background:#f4f4f4;border:1px solid #ddd;'
                  + 'padding:8px 10px;margin:6px 0 0;white-space:pre-wrap;';
out.textContent = 'Drag the scene. The camera of the current view appears here; '
                + 'paste that line over CAMERA in the cell.';
gd.parentNode.insertBefore(out, gd.nextSibling);
function f(v) { return Number(v).toFixed(3); }
gd.on('plotly_relayout', function (ev) {
  var cam = ev['scene.camera'];
  if (!cam || !cam.eye) { return; }
  var s = 'CAMERA = dict(eye=dict(x=' + f(cam.eye.x) + ', y=' + f(cam.eye.y) + ', z=' + f(cam.eye.z) + ')';
  if (cam.up)     { s += ', up=dict(x=' + f(cam.up.x) + ', y=' + f(cam.up.y) + ', z=' + f(cam.up.z) + ')'; }
  if (cam.center) { s += ', center=dict(x=' + f(cam.center.x) + ', y=' + f(cam.center.y) + ', z=' + f(cam.center.z) + ')'; }
  out.textContent = s + ')';
});
"""
EPS_NAME = {0.007843: "2/255", 0.015686: "4/255", 0.031373: "8/255"}
SYMBOL = {0.007843: "circle", 0.015686: "square", 0.031373: "diamond"}
PANE_LEFT, PANE_BACK, PANE_FLOOR = "#EDEDED", "#E4F0FA", "#FBE9EA"
_C = ["I(X;T)-In", "I(T;Y)-In"]

# ---- MI of every trajectory checkpoint -------------------------------
mi = pd.read_csv(CSV_MI)
mi = mi[(mi.run_tag == RUN_TAG) & (mi.bins == BINS) & (mi.in_size == IN_SIZE)].copy()
mi["key"] = mi.at_scenario + "||" + mi.epoch.astype(int).astype(str)

# ---- the baseline's metric for the same checkpoints ------------------
bl = pd.read_csv(CSV_BASE.format(BASELINE))
traj = bl[bl.Case_Set == "trajectory"].copy()
traj["key"] = (traj.Model_Dir.str.split("/", n=1).str[1] + "||"
               + traj.Epoch.astype(int).astype(str))
df = mi.merge(traj[["key", "Family", "AT_Eps", METRIC]], on="key", how="inner")
df = df[df.Family == FAMILY].sort_values(["AT_Eps", "epoch"])
if EPS is not None:
    df = df[np.isclose(df.AT_Eps.astype(float), EPS)]
if df.empty:
    raise ValueError(f"no rows for family {FAMILY!r}, eps {EPS!r} in {BASELINE}.csv")
eps_title = EPS_NAME.get(round(float(EPS), 6), EPS) if EPS is not None else "2/255, 4/255, 8/255"

tau = TAU.get((BASELINE, METRIC))
fig = go.Figure()

# ---- the pre-AT source: where every trajectory starts ----------------
pre_csv, pre_name = CSV_PRE[FAMILY]
pre = pd.read_csv(pre_csv)
pre = pre[(pre.model_name == pre_name) & (pre.bins == BINS) & (pre.in_size == IN_SIZE)]
pre_b = bl[(bl.Case_Set == "sources") & (bl.Family == FAMILY)]
pxyz = None
if len(pre) and len(pre_b):
    pxyz = (float(pre[_C[0]].iloc[0]), float(pre[_C[1]].iloc[0]),
            float(pre_b[METRIC].iloc[0]))
    fig.add_trace(go.Scatter3d(
        x=[pxyz[0]], y=[pxyz[1]], z=[pxyz[2]], mode="markers",
        marker=dict(size=11, color="#2CA02C", symbol="diamond",
                    line=dict(color="black", width=1.2)),
        name=f"Suspect before AT   {METRIC}={pxyz[2]:.2f}"))

# ---- one trajectory per eps ------------------------------------------
for i, (eps, g) in enumerate(df.groupby("AT_Eps")):
    g = g.sort_values("epoch")
    key = round(float(eps), 6)
    fig.add_trace(go.Scatter3d(
        x=g[_C[0]], y=g[_C[1]], z=g[METRIC], mode="lines+markers",
        line=dict(color="#9AA0A6", width=2),
        marker=dict(size=6, symbol=SYMBOL.get(key, "circle"),
                    color=g.epoch, colorscale="Viridis", cmin=0, cmax=29,
                    line=dict(color="white", width=0.5),
                    colorbar=dict(title="AT epoch", x=1.02, len=0.62,
                                  thickness=14) if i == 0 else None,
                    showscale=(i == 0)),
        name=f"AT eps {EPS_NAME.get(key, eps)}"))
    # the first AT epoch, dotted back to where it started
    if pxyz is not None:
        fig.add_trace(go.Scatter3d(
            x=[pxyz[0], g[_C[0]].iloc[0]], y=[pxyz[1], g[_C[1]].iloc[0]],
            z=[pxyz[2], g[METRIC].iloc[0]], mode="lines",
            line=dict(color="#9AA0A6", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip"))

# ---- the decision threshold, as a plane with a drawn edge ------------
xs = [df[_C[0]].min(), df[_C[0]].max()]
ys = [df[_C[1]].min(), df[_C[1]].max()]
if pxyz is not None:
    xs = [min(xs[0], pxyz[0]), max(xs[1], pxyz[0])]
    ys = [min(ys[0], pxyz[1]), max(ys[1], pxyz[1])]
px, py = 0.04 * (xs[1] - xs[0]), 0.04 * (ys[1] - ys[0])
xs, ys = [xs[0] - px, xs[1] + px], [ys[0] - py, ys[1] + py]
if tau is not None:
    gx, gy = np.meshgrid(xs, ys)
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=np.full_like(gx, tau), showscale=False, opacity=0.22,
        colorscale=[[0, "#D62728"], [1, "#D62728"]], hoverinfo="skip",
        name=f"tau = {tau}", showlegend=True))
    fig.add_trace(go.Scatter3d(
        x=[xs[0], xs[1], xs[1], xs[0], xs[0]],
        y=[ys[0], ys[0], ys[1], ys[1], ys[0]], z=[tau] * 5, mode="lines",
        line=dict(color="#D62728", width=3, dash="dash"),
        showlegend=False, hoverinfo="skip"))

AX = dict(showbackground=True, gridcolor="white", zeroline=False,
          showspikes=False, linecolor="#9AA0A6", tickfont=dict(size=10),
          title=dict(font=dict(size=13)))
fig.update_layout(
    title=(f"{FAMILY} · post-hoc adversarial training, eps {eps_title} · {BASELINE} {METRIC}<br>"
           f"<sub>floor = information plane on group_A · height = {BASELINE} "
           f"{METRIC} · colour = AT epoch 0→29 · dotted = the first AT epoch</sub>"),
    scene=dict(
        xaxis=dict(AX, title="I(X;T)", backgroundcolor=PANE_LEFT),
        yaxis=dict(AX, title="I(T;Y)", backgroundcolor=PANE_BACK),
        zaxis=dict(AX, title=f"{BASELINE}  {METRIC}", backgroundcolor=PANE_FLOOR,
                   **({"range": list(Z_RANGE)} if Z_RANGE else {})),
        camera=CAMERA or dict(eye=dict(x=1.65, y=-1.75, z=0.75)),
        aspectratio=dict(x=1.1, y=1.1, z=0.85)),
    paper_bgcolor="white", width=1020, height=720,
    legend=dict(x=0.60, y=0.97, bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#CCCCCC", borderwidth=1, font=dict(size=11)))

if INTERACTIVE:
    # The readout is a post_script, and only the HTML renderers run one. The
    # JupyterLab / VS Code plotly extensions render through the mimetype path,
    # which drops it silently -- the figure appears, the readout does not. So
    # force an HTML renderer here, and also write a standalone HTML that
    # certainly works when opened in any browser.
    fig.show(renderer="notebook_connected", post_script=CAPTURE_CAMERA_JS)
    _html = "./figures/traj3d_camera_capture.html"
    import os
    os.makedirs(os.path.dirname(_html), exist_ok=True)
    fig.write_html(_html, include_plotlyjs="cdn", post_script=CAPTURE_CAMERA_JS)
    print(f"camera readout also available at {os.path.abspath(_html)}  (open in a browser)")
else:
    fig.show()
