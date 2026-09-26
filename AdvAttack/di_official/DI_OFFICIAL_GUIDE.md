# Dataset Inference（官方代码路径）Guide

用 **官方 repo 的代码** 跑 Dataset Inference（Maini et al., ICLR 2021）黑盒协议（Blind Walk），
只有模型对象和图片集合来自本框架。目的：无论结果好坏，都可以说"评测用的是官方代码，没有系统性改动"。

相关文件：
- 入口（YAML 批量）：`main_DI_official_eval.py`
- 适配层：`AdvAttack/di_official/di_official_adapter.py`
- 官方代码（vendored）：`AdvAttack/di_official/di_vendored/`
  - `attacks.py`：官方文件的字节级拷贝，仅两行 torch-2.x 兼容补丁（`SOURCE.md` 有 diff 和 sha256）
  - `generate_features.py`：官方 `get_random_label_only` 逐字拷贝
  - `notebook_protocol.py`：官方 `CIFAR10_rand.ipynb` 的 cell 逐字转写 + `notebooks/utils.generate_table`
  - `SOURCE.md`：来源 commit、哈希、全部改动清单、保留的官方 quirk
- 计划文件：`saved_exp_plan/di_official_plan/*.yaml`
- 旧的自实现版本 `AdvAttack/DI.py` / `main_DI_eval.py` 原样保留，互不影响。

---

## 1. 官方流程与本实现的对应

| 官方 repo 步骤 | 官方代码 | 这里的调用 |
|---|---|---|
| 训练 victim / suspects | `train.py`（WRN-28-10 等） | 不用；模型来自本框架的 checkpoint，套 `NormalizedModel`（= 官方 `--normalize 1`，归一化在模型内部） |
| 取 1000 张私有 + 1000 张公共图片 | `funcs.get_dataloaders(train_shuffle=False)`，无增强 | `raw_train_clean_set` 上 group_A 子集前 1000 张 + `raw_test_set` 前 1000 张，`make_loader`（不打乱、无增强） |
| Blind Walk 特征 | `generate_features.py --feature_type rand` → `get_random_label_only` → `attacks.rand_steps` | 同名函数，vendored，逐字 |
| 特征落盘 | `files/{DATASET}/model_{name}_normalized/{train,test}_rand_vulnerability.pt` | 同一布局，root = `saved_logs/di_official/files/victim=<arch>_<id>/` |
| 标准化 + 整形 | notebook cell 8-9（`mean/std` 来自 teacher 私有特征；`.T.reshape(1000,30)`） | `notebook_protocol.load_features / normalize_and_flatten` |
| 回归器 | cell 10-12：Linear(30,100)-ReLU-Linear(100,1)-Tanh，SGD 0.1，全批 1000 轮，前 `split_index=500` 行 | `build_regressor_training_set / build_regressor / train_regressor` |
| 打分与检验 | cell 15-18：后 500 行，单侧 Welch（public > private） | `score_features / hold_out / inference_stats` → `P_Value_Welch` |
| 论文表格协议 | `utils.generate_table(selected_m=10)`：每侧抽 10 行、100 次、p 值调和平均 | 同名函数 → `P_Value_M` |

victim 在协议里必须叫 `teacher`（notebook 用 `trains["teacher"]` 算标准化统计量并训练回归器），适配层自动处理。

## 2. 改动清单

全部列在 `di_vendored/SOURCE.md`。概括：

- `attacks.py`：`remaining[remaining] = new_remaining` → `remaining[remaining.clone()] = new_remaining`（两处）。torch ≥ 1.10 禁止用被写张量自身做掩码的原地写入，clone 掩码与 1.8 语义完全一致。
- `generate_features.py`：只保留 `get_random_label_only`；顶层 import 改为相对导入；`device` 由适配层注入（官方只在 `__main__` 里赋值）。
- notebook 转写：`names` / 文件根目录参数化；`.T` → `_reverse_dims`（等价）；`tqdm.notebook` → `tqdm.auto`；`ipdb.set_trace()` → `raise`；`reshape(1000, …)` 的 1000 → `num_images` 参数（默认 1000）；cell 20 去掉 `to_hdf`；新增 `inference_stats` 返回 `print_inference` 打印的数值。
- 环境垫片：`ipdb` 占位模块；`generate_features.device` 注入；特征调用外层套 `torch.no_grad()`（官方在 `rand_steps` 内部已是 no_grad，外层两次前向不用梯度）。
- 适配层新增的行为：每个模型抽特征前重置种子（官方只种 torch，不种 numpy；这里两者都种）；`generate_table` 每个模型前重置种子，使每一行不依赖同批其他模型（`per_name_reseed`，可关）。

保留不动的官方 quirk（`SOURCE.md` 有解释）：`.T.reshape` 造成的"每行 = 30 张连续图片、单一范数"布局；未翻转的走法终止于 51 倍噪声；文件名 `_vulnerability.pt`。

## 3. 运行

```powershell
Set-Location E:\Experiment
# 全部计划
& 'C:\Users\louj\python_env\pytorch_env\Scripts\python.exe' -X utf8 -u .\main_DI_official_eval.py
# 单个计划
& 'C:\Users\louj\python_env\pytorch_env\Scripts\python.exe' -X utf8 -u .\main_DI_official_eval.py --yaml .\saved_exp_plan\di_official_plan\CIFAR10_RES18_DIofficial_NegRes.yaml
# 跑完所有计划后，把该 victim 下所有缓存模型放进同一次协议重新打分（不重抽特征）
& 'C:\Users\louj\python_env\pytorch_env\Scripts\python.exe' -X utf8 -u .\main_DI_official_eval.py --yaml .\saved_exp_plan\di_official_plan\CIFAR10_RES18_DIofficial_NegRes.yaml --protocol-only
```

默认参数即官方值：`--num-images 1000 --batch-size 500 --split-index 500 --feature-seed 0 --protocol-seed 0 --selected-m 10 --inner-rep 100 --regressor-epochs 1000 --alpha 0.05`。

现有计划（victim = CIFAR-10 ResNet-18 seed 42）：

| 文件 | 内容 |
|---|---|
| `CIFAR10_RES18_DIofficial_NegRes.yaml` | ResNet-18 独立池，seed 42–51 × overlap {0.0, 1.0}，共 19 个（seed 42 / 1.0 就是 victim，已剔除） |
| `CIFAR10_RES18_DIofficial_NegVGG.yaml` | VGG16 独立池，20 个 |
| `CIFAR10_RES18_DIofficial_Pos_FT_AT.yaml` | 18 个 FT / AT 逃逸阳性（同 `at_eval_plan/CIFAR10_RES18_ROBD_FT_at.yaml`） |
| `CIFAR10_RES18_DIofficial_Pos_Extraction.yaml` | 7 个 extraction 阳性（RN18 / VGG16 / DeiT，同 `di_eval_plan`） |

YAML 语法与 `main_DI_eval.py` 相同（`Victim` + `Positive` 或 `Negative Suspect`，`Model_Path` 相对 `saved_models/`）。新增 suspect 只需加路径；已缓存的特征不会重抽（`--force-features` 强制）。

缓存布局（可直接喂给官方 notebook，见第 6 节）：

```
saved_logs/di_official/files/victim=ResNet-18_CIFAR-10_25000_seed=42_overlap=1.0/CIFAR10/
    model_teacher_normalized/{train,test}_rand_vulnerability.pt + meta.json
    model_<arch>__<scenario前48字符>__<hash8>_normalized/...
```
`meta.json` 记录完整 scenario、checkpoint 路径与 sha256、私有/公共索引、种子、耗时、vendored attacks.py 的 sha256。

## 4. 主表 `saved_logs/di_official/master.csv`

每个模型一行（victim 自身也有一行，`Suspect_Type=victim`，作为阳性上限）。关键列：

| 列 | 含义 |
|---|---|
| `Mean_Diff_Welch`, `P_Value_Welch` | notebook cell 16/18：后 500 行，公共分数均值 − 私有分数均值，单侧 Welch p 值 |
| `Mean_Diff_M`, `P_Value_M` | `generate_table(selected_m=10)`：论文表格用的协议 |
| `Stolen_Welch`, `Stolen_M` | **本脚本**对 p 值取 `< Alpha` 的判定；官方只打印 p 值并画 0.05 / 0.01 线 |
| `Num_Images … Regressor_Epochs` | 本行使用的全部协议参数；去重键包含它们，改参数会生成新行而不是覆盖 |

同一 victim 的回归器只依赖 teacher 特征文件和 `protocol_seed`，所以不同批次跑出的行互相可比；`--protocol-only` 重跑会得到逐位相同的数字（smoke test 已验证）。

## 5. 结果解读的前提

- DI 的 H0 是"嫌疑模型没有用 victim 的私有数据训练"。**overlap 1.0 的负样本与 victim 共享 group_A，按 DI 的定义就应该被判阳性**；只有 overlap 0.0 对应官方的 independent。报告时两者必须分开，否则对 DI 不公平。
- 官方的 independent 模型是在 CIFAR-10 测试集上训练的，它的"公共"点正是自己的训练数据，因此 mean_diff 为负、判负容易；我们的 overlap 0.0 负样本对私有和公共图片都没见过，是更严格的 H0。
- `.T.reshape` 的整形使回归器看到的不是逐图 30 维向量而是"30 张连续图片在单一范数下的距离"，训练行是全部 L_inf 行加部分 L2 行，评估行是其余 L2 行加全部 L1 行。这是官方代码的实际行为，保留是为了"官方代码"这一主张；若要论文文字描述的逐图布局，需另加开关，不在此路径内。
- 私有/公共两组图片的类别构成会影响任何模型的距离分布。n=1000 时两组每类相差 ≤ 23 张（约 2%），n=100 的 smoke 数值不可用于结论。

## 6. 用官方 notebook 复算

把 `saved_logs/di_official/files/victim=.../` 作为 `root_path`，把 `CIFAR10_rand.ipynb` 第二个参数 cell 里的 `root_path` 改成它、`names` 改成 `["teacher", <各 model_ 目录名去掉前后缀>]`，`generate_table` 的 `order` 传同一列表即可逐格运行；官方 notebook 需要的 `torch/scipy/pandas/seaborn` 本环境都有（`to_hdf` 那一格需要 PyTables，可跳过）。

## 7. 耗时（RTX 5090，victim = ResNet-18）

官方设置（1000 张 / batch 500 / 每张 30 条走法 / 至多 50 步）下：

| 模型 | 公共集抽特征 | 私有集抽特征 |
|---|---|---|
| ResNet-18 vanilla | 34–37 s | 36–37 s |
| ResNet-18 对抗训练微调 | 52 s | 54 s |

即每个模型约 1.2–1.8 分钟；协议部分（回归器 1000 轮 + generate_table 100 次抽样）每个模型秒级。
NegRes 19 个约 25 分钟，NegVGG 20 个约 25 分钟，Pos_FT_AT 18 个约 30 分钟，Pos_Extraction 7 个约 10 分钟（DeiT 略慢）。

## 7b. 未翻转走法的诊断

官方 `rand_steps` 每次调用打印一行 `Number of steps = … | Failed to convert = N`，N 是这个 batch 的 500 张图里沿**这一个方向**没有翻转的张数；每种范数 10 个方向就是 10 行。适配层把这些行原样打印到控制台的同时解析它们，把每种范数的未翻转总数写进 `meta.json` 的 `walks_private` / `walks_public`（`unflipped`、`unflipped_frac`、`steps_printed`）。

汇总用 `python summarize_di_official_walks.py`：扫描 `saved_logs/di_official/files` 下所有 victim 与 `fseed=` / `steps=` 子目录，按模型、按私有/公共、按范数给出未翻转比例、第 1 步即翻转（模型本来分错）比例、平均距离，写到 `saved_logs/di_official/walk_summary.csv`。`src` 列为 `exact` 表示来自解析的计数；`threshold` 表示旧缓存没有计数、用"距离 ≥ 0.95 × 最大值"估计，50 步预算下可靠，更大预算下 L2/L1 会低估。

## 8. 注意事项

- `--batch-size` 必须整除 `--num-images`（官方用 `i+1 >= num_images/batch_size` 退出循环，否则多出样本、notebook `reshape` 报错）。
- 官方 `attacks.py` 在 import 时创建 CUDA 张量，`rand_steps` 的高斯噪声硬编码 `.cuda()`，必须有 GPU 且模型在 `cuda:0`。
- 特征目录名带 48 字符截断 + 8 位哈希，是为了 Windows 260 字符路径上限；完整 scenario 在 `meta.json` 和主表里。
- `set_seed(..., deterministic=True)` 与官方走法兼容（掩码 index_put 在确定性模式下可用）；如遇不确定性算子报错可加 `--nondeterministic`。
- 每个模型抽特征前会重置种子：公共集先抽、私有集后抽，与官方 `feature_extractor` 的顺序一致，所以所有模型用同一批噪声方向。
- **噪声方向的随机性官方不做重复**（每个模型只抽一次特征，torch seed 0）。要量化它就换 `--feature-seed k` 重跑整条流程（teacher 和 suspect 都重抽）。非零种子的特征自动落到 `victim=.../fseed=k/` 子目录，主表按 `Feature_Seed` 生成新行；缓存的 meta.json 与请求的种子、图片数、batch 不一致时会直接报错，不会误用旧特征。`--protocol-seed` 只影响行抽样，秒级，可以随意扫。

---

## 验证记录（2026-09-10）

Smoke test（100 张、batch 50、split 50）：vendored `attacks.py` 与上游的 diff 恰为两行补丁；`_reverse_dims` 与 `.T` 逐元素相等；
`--protocol-only` 重跑得到逐位相同的数字；特征形状 `[N, 10, 3]` 与上游一致。

全尺寸运行（官方默认参数，`tmp/di_official_validate*.yaml`，日志 `tmp/di_official_validate.log`，4 个模型共 5.5 分钟）：

| 模型 | 类型 | mean_diff (Welch) | p (Welch, 500 行) | mean_diff (m=10) | p (m=10, 100 次调和平均) |
|---|---|---|---|---|---|
| victim 自身 | victim | 1.622 | 1.9e-316 | 1.605 | 3.9e-30 |
| RN18 seed 43, overlap 1.0（同数据独立训练） | negative | 1.486 | 2.4e-257 | 1.507 | 4.2e-15 |
| RN18 seed 42, overlap 0.0（不相交数据） | negative | 0.569 | 1.3e-29 | 0.591 | 1.1e-3 |
| FT-AL PGD eps 0.031 微调阳性 | positive | 0.291 | 1.8e-17 | 0.323 | 5.3e-2 |

对照官方 notebook 里 CIFAR-10 rand 的数字：teacher mean_diff 1.823、independent −0.397。这里 victim 的 1.62 与官方量级一致；
overlap 1.0 的负样本被强烈判阳，符合 DI 检测"数据使用"的定义。需要注意的是 overlap 0.0 的负样本（私有和公共图片都没见过）
在官方协议下也被判阳（两种 p 值都小于 0.05），说明官方的 Blind Walk 回归器在"同分布、不相交数据"的 H0 下不是零信号；
官方的 independent 之所以 p≈1，是因为它在测试集上训练，公共点反而是它的训练数据。这一条需要跑完整个 overlap 0.0 池（10 个 seed）后再下结论。
私有与公共图片的输入变换已核对为完全相同（均仅 ToTensor）。
