"""
audit_ipguard_ondisk.py

对磁盘上已生成的 IP-Guard 指纹数据做"生成后自检":在【生成这些指纹的 victim
模型】上重算 Eq.(5) 的目标函数,逐点判定每个指纹点是否满足优化要求。

Eq.(5) 目标 = ReLU(z_i - z_j + k) + ReLU(max_{t!=i,j} z_t - z_i)，收敛时应为 0：
  term1 = 0  <=>  z_j >= z_i + k        (已跨过 i->j 边界，目标类 j 领先 margin k)
  term2 = 0  <=>  z_i >= 其它所有类      (源类 i 稳居第二，无第三类干扰)
两者都满足 <=> logit 排序为 z_j >= z_i >= 其余，argmax = j。

关键前提：必须用 victim（生成模型）审查，不能用 suspect —— "满足优化要求"是相对
生成模型定义的。审查同时也是对保存的 `converged` 标志的独立复核。

用法:  python audit_ipguard_ondisk.py
"""

import re
import sys
from pathlib import Path

import torch
import pandas as pd

from Model.ResNet_18 import ResNet18
from main_adv_tra import NormalizedModel
from AdvAttack.IP_Guard import IPGuardGenerator

try:
    sys.stdout.reconfigure(encoding="utf-8")   # 允许中文输出（Windows 控制台）
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 配置：victim 模型 + 数据集归一化 + 指纹根目录 ----
VICTIM_CKPT = "saved_models/vanilla/CNN_Models/CIFAR-10_ResNet-18_25000_42_1.0/best_epoch.pth"
MEAN, STD   = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
NUM_CLASSES = 10
IPGUARD_ROOT = Path("Indices/CIFAR-10/IPGuard")
CONFIGS = ["TR", "TL", "RR", "RL"]
TOL = 1e-4


def load_victim():
    net = ResNet18(num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(VICTIM_CKPT, map_location=device))
    return NormalizedModel(net, MEAN, STD).to(device).eval()


def audit_ipguard(model, fp, k, tol=TOL):
    """逐点重算 Eq.(5)，返回 (per_point_df, summary_dict)。"""
    X = fp["points"].float().to(device)
    i = torch.as_tensor(fp["source_labels"]).long()
    j = torch.as_tensor(fp["target_labels"]).long()
    conv = torch.as_tensor(fp["converged"]).bool()

    with torch.no_grad():
        logits = model(X).cpu()
    N = logits.shape[0]
    ar = torch.arange(N)
    z_i, z_j = logits[ar, i], logits[ar, j]

    masked = logits.clone()
    masked[ar, i] = -float("inf")
    masked[ar, j] = -float("inf")
    z_oth = masked.max(1).values

    term1 = torch.relu(z_i - z_j + k)     # 想要 0
    term2 = torch.relu(z_oth - z_i)       # 想要 0
    obj = term1 + term2
    pred = logits.argmax(1)
    box_ok = bool(X.min() >= -1e-6 and X.max() <= 1 + 1e-6)

    per_point = pd.DataFrame({
        "i": i.numpy(), "j": j.numpy(), "pred": pred.numpy(),
        "pred==j": (pred == j).numpy(),
        "z_j - z_i": (z_j - z_i).numpy().round(4),
        "term1": term1.numpy().round(6),
        "term2": term2.numpy().round(6),
        "objective": obj.numpy().round(6),
        "satisfies": (obj <= tol).numpy(),
        "conv_flag": conv.numpy(),
    })
    summary = {
        "N": N,
        "pred_eq_j": int((pred == j).sum()),
        "term1_ok": int((term1 <= tol).sum()),
        "term2_ok": int((term2 <= tol).sum()),
        "satisfies": int((obj <= tol).sum()),
        "conv_flag": int(conv.sum()),
        "agree_w_conv": int(((obj <= tol) == conv).sum()),
        "box_ok": box_ok,
        "max_obj": round(float(obj.max()), 6),
    }
    return per_point, summary


def main():
    victim = load_victim()
    print(f"victim: {VICTIM_CKPT}")
    print(f"scanning: {IPGUARD_ROOT}\n")

    rows, unsatisfied = [], []
    for d in sorted(IPGUARD_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = re.search(r"k=([0-9.]+)", d.name)
        if m is None:
            print(f"[skip] cannot parse k from dir name: {d.name}")
            continue
        k = float(m.group(1))
        for tag in CONFIGS:
            fp_path = d / f"{tag}.pt"
            if not fp_path.exists():
                continue
            fp = IPGuardGenerator.load_fingerprints(fp_path)
            per_point, summ = audit_ipguard(victim, fp, k)
            rows.append({"dir": d.name, "k": k, "cfg": tag, **summ})
            bad = per_point[~per_point["satisfies"]]
            if len(bad):
                unsatisfied.append((d.name, tag, bad))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    cols = ["k", "cfg", "N", "pred_eq_j", "term1_ok", "term2_ok",
            "satisfies", "conv_flag", "agree_w_conv", "box_ok", "max_obj"]
    # 目录太长，单独打印一次映射再用短名
    dirs = df["dir"].unique()
    short = {name: f"D{idx}" for idx, name in enumerate(dirs)}
    for name, s in short.items():
        print(f"  {s} = {name}")
    print()
    df["D"] = df["dir"].map(short)
    print(df[["D"] + cols].to_string(index=False))

    tot_ok, tot_n = int(df.satisfies.sum()), int(df.N.sum())
    tot_agree = int(df.agree_w_conv.sum())
    print(f"\nTOTAL: satisfies = {tot_ok}/{tot_n} | "
          f"agree_with_conv_flag = {tot_agree}/{tot_n}")

    if unsatisfied:
        print(f"\n[!] {len(unsatisfied)} file(s) contain unsatisfied points:")
        for name, tag, bad in unsatisfied:
            print(f"  {name}/{tag}: {len(bad)} points")
            print(bad.to_string())
    else:
        print("\nAll fingerprint points satisfy the Eq.(5) requirement on the victim.")


if __name__ == "__main__":
    main()
