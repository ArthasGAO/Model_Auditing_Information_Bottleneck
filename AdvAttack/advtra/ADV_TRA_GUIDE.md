# ADV-TRA 指纹:生成与评估 Guide

轨迹式(trajectory-based)模型指纹。一次性在 **victim** 上刻出一批"跨越决策边界的轨迹"存盘;
之后对任意 **suspect** 模型回放这些轨迹,用预测一致性判断是否盗用。

相关文件:
- 入口(YAML 批量):`main_adv_tra.py` → `main_adv_tra_neg` / `main_adv_tra_pos`
- 适配层(桥接本框架与作者原码):`AdvAttack/advtra/adv_tra_adapter.py`
- 作者原码(vendored):`AdvAttack/advtra/advtra_vendored/adv_gen.py`

---

## 1. 生成过程(extraction)

### 1.1 流程
1. 载入 **victim** 作为 source model,包成 `NormalizedModel`(接收 raw [0,1] 输入)。
2. 选 base 图像:`base_indices = train_subset_indices[:2*num_trajectories]`,来自
   `Indices/CIFAR-10/group_A_subset_10000_from_25000_seed42.npy`。
   **group_A 就是 victim 训练用的数据**,所以起点都是 victim 见过的训练样本。
   给 `2*num_trajectories`(=200)个候选是留冗余,因为部分样本会生成失败。
3. `write_data_log` 把选中的 (X_train, y_train) 写成 `data_log.pth`。
4. Monkey-patch:把 vendored 的 `build_model` 换成返回已加载好的 victim(经 `_ModelShim`),
   于是原码的 `load_state_dict` / `.to` 变成空操作,`suspect_path` 只是占位。
5. `generate_trajectory(args)` 逐个 base 样本尝试;成功刻出完整轨迹的才保存,直到攒够
   `num_trajectories` 条。失败样本(过不了 boundary probing)被跳过,不占编号。

### 1.2 一条轨迹的结构(81 点 = 9 hop × 9 点)
- **1 条轨迹 = 1 张 base 图像**(不同轨迹来自不同图像)。
- 起始类 `c_0` = base 图像的真实类;再随机抽 `tra_classes-1 = 9` 个其它类作为访问顺序
  ([adv_gen.py](advtra_vendored/adv_gen.py) `generate_all_classes`)。
- **每个 hop = 对一条边界的 9 点双侧跨越**:
  - `generate_unilateral_tra`:优化 `length/2 = 4` 个步长,做 FGSM 式 signed-gradient
    步进逼近 target。约束:第 4 点必须跨到 target(status1),更早的点不许跨(status2),
    连续步长几何衰减(`factor_re=0.95`,越靠边界步子越小)。
  - `generate_bilateral_tra`:把 4 个步长镜像(`s + s[::-1]`)→ 8 步 → 9 点(1 起点 + 8 步)。
- **链式拼接**:每个 hop 的最后一点 = 下一个 hop 的起点([adv_gen.py](advtra_vendored/adv_gen.py) `image = tra[-1]`)。
  → 相邻 hop 共享端点,81 点里只有 **73 个不同图像**。
- **第 0 点(hop0/pos0)= clean base 图像**,未加任何扰动;其余 80 点都是合成的。

### 1.3 存盘格式
路径:`{fingerprint_path}/{dataset}/trajectory_{length}/{1..N}/`
每个轨迹目录两个文件:
- `tra_log.pth`:9 个 hop 张量的 **list**,每个 `(length+1, 3, 32, 32)` float32,**raw [0,1]**。
  `torch.cat` 后 = `(81, 3, 32, 32)`。
- `pred_log.pth`:9 个张量的 list,拼接后 `(81,)` = **victim 在每个点上的预测**(答案键)。

> 注意是嵌套 list(按 hop),不是单一扁平张量;用时 `torch.cat(...)` 展平。

### 1.4 关键超参(`main_adv_tra.py` 当前取值)
| 参数 | 值 | 含义 |
|---|---|---|
| `length` | 8 | 每个 hop 步数;`half_length=4`,9 点/hop。**决定存盘目录 `trajectory_8`** |
| `tra_classes` | 10 | 访问的类数 → 9 个 hop |
| `num_trajectories` | 100 | 目标轨迹数 |
| `max_iteration` | 1000 | 单 hop 步长优化上限 |
| `initial_stepsize` / `tra_lr` | 0.05 / 0.05 | 初始步长 / 步长优化学习率 |
| `factor_lc` | 0.9 | 长度控制因子 |
| `factor_re` | 0.95 | 步长衰减(刹车)因子 |
| `threshold` | 0.5 | 验证阈值(见第 3 节) |

### 1.5 CIFAR-10 现存指纹
`results/advtra1/CIFAR-10_ResNet-18_25000_42_1.0/fingerprints/cifar10/trajectory_8/`
- victim = `CIFAR-10_ResNet-18_25000` seed42,100 条轨迹,length=8。
- 起始类覆盖全部 10 类(每类 7–12 条,近似均匀)。
- 据 `extraction_log.txt`:~110 次尝试、6 次失败 → 100 条成功保存。
- **可直接复用,无需重跑 extraction。**

---

## 2. 只想跑生成一次(新 victim 时)
```python
from AdvAttack.advtra.adv_tra_adapter import build_args, run_extraction
args = build_args(dataset_name="cifar10", num_classes=10,
    data_path=..., model_path=..., fingerprint_path=...,
    num_trajectories=100, length=8, tra_classes=10, threshold=0.5, device=device)
run_extraction(args, wrapped_source_model=victim, raw_dataset=raw_train_clean_set,
               base_indices=train_subset_indices[:200].tolist())
```
`run_extraction` 会在结束后把 `args.num_trajectories` 同步成磁盘上真实成功的条数。

---

## 3. 评估过程(verification)

### 3.1 两级聚合(核心公式,见 [adv_gen.py](advtra_vendored/adv_gen.py) `verify_trajectory`)
对每条轨迹:
```
suspect 对 81 点预测 tra_pred;victim 存下的预测 ori_pred(pred_log)
mutation = mean(tra_pred != ori_pred)          # 该轨迹的“突变率”
若 mutation < threshold(0.5): 记一次“识别成功”
```
对整个模型:
```
detection_rate = 识别成功的轨迹数 / num_trajectories
```
- 盗用模型边界像 victim → 预测一致 → mutation 低 → detection 高。
- 独立模型边界不同 → mutation 高 → detection 低。
- 最终判定 stolen:`detection_rate > threshold`(在 adapter/CSV 层再比一次)。
- **threshold=0.5 被复用两次**:单条轨迹的 mutation 阈值 + 整体 detection 的 stolen 阈值。

### 3.2 跑单个模型(最小代码)
```python
from pathlib import Path
import torch
from AdvAttack.advtra.adv_tra_adapter import build_args, run_verification_pretty
from Model.ResNet_18 import ResNet18
from main_adv_tra import NormalizedModel

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
ROOT = Path("./results/advtra1/CIFAR-10_ResNet-18_25000_42_1.0")

args = build_args(dataset_name="cifar10", num_classes=10,
    data_path=str(ROOT/"data"), model_path=str(ROOT/"_model_paths"),
    fingerprint_path=str(ROOT/"fingerprints"),
    num_trajectories=100, length=8, tra_classes=10, threshold=0.5, device=device)

net = ResNet18(num_classes=10)
net.load_state_dict(torch.load("<被测模型.pth>", map_location=device))
net = NormalizedModel(net, MEAN, STD).to(device).eval()

result = run_verification_pretty(args, wrapped_suspect_model=net,
            suspect_path=str(ROOT/"_model_paths/cifar10/_probe.pth"))
print(result)   # detection_rate / mean_mutation_rate / num_trajectories / threshold
```

### 3.3 批量入口
- `main_adv_tra_neg`:遍历独立训练的负样本(seed 42–51 × overlap),期望 detection 低。
- `main_adv_tra_pos`:遍历 YAML `Positive.Model_Path` 里的盗用模型,期望 detection 高。
- 都先复用/生成指纹,再对每个 suspect 调 `run_verification_pretty`,结果写入
  `saved_logs/at_eval/ADV_TRA/{Negative,Positive}/*.csv`。

---

## 4. 容易忘的坑
- **`length` 和三个 path 必须与 extraction 时一致**,否则找不到轨迹文件直接报错。
- **`num_trajectories` 会被自动纠正**成磁盘真实条数(`_count_saved_trajectories`),写多写少不崩。
- **`suspect_path` 只是占位**:真实模型经 shim 注入,该文件内容被忽略,只需可写路径。
- **模型必须 raw [0,1] 输入**(`NormalizedModel` 包好),因为轨迹点是 [0,1] 图像。
- **生成只约束 target 类**:中间点是否路过第三类既不检查也不阻止,只如实记进 `pred_log`。
  所以一个 hop 内可能出现 >1 次 victim 预测翻转(轨迹经过多类交界区)。

---

## 5. 参考基线(本项目实测,victim=CIFAR-10 ResNet-18 25000 的指纹)
| suspect | detection_rate | mean_mutation | 判定 |
|---|---|---|---|
| FT(fine-tune 盗用) | 1.00 | 0.09 | ✅ stolen |
| Knockoff(抽取盗用) | 0.12 | 0.68 | ❌ 漏检(可规避) |
| source vs source(sanity) | ≈1.00 | ≈0 | 自检,必须接近 1 |

要点:ADV-TRA 对 fine-tune 盗用几乎完美,但对 **Knockoff 抽取模型漏检**——因为抽取模型
边界几何与 victim 差异大,victim 上刻的轨迹在其上预测频繁偏离(mutation 高)。这与
boundary-band 分析一致:同一"边界不同"的事实,既让 band gate 觉得指纹贴近抽取模型边界,
也让 ADV-TRA 认不出它。

---

## 6. 逐点审查工具(调试/可视化)
把一条轨迹在某模型上的一阶边界距离逐点看清(delta1 / (k,j) / ill),见
`AdvAttack/boundary_band.py::collect_boundary_stats_batched`(传 `(X, victim_pred)` 元组时,
victim_pred 进 `label` 列,于是 `y_pred==label` 即 "与 victim 一致" = 1−mutation)。
沿链的 delta1 呈 V 形(谷底 ≈ 每个 hop 的跨越点 pos 3–4),`(k,j)` 在谷底两侧对调,
可作为轨迹构造意图的独立验证。
