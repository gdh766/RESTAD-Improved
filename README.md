# RESTAD 改进版：基于重建与相似度 Transformer 的时间序列异常检测

本科毕业设计成果。本仓库基于论文 **RESTAD: REconstruction and Similarity based Transformer for
time series Anomaly Detection**（Ghorbani, Reinders, Tax；arXiv:2405.07509）的
[官方实现](https://github.com/Raminghorbanii/RESTAD) 完成复现，并在此基础上做了 5 处改进。

- 原作者代码与说明见 [`README_original.md`](README_original.md)
- 论文中文译文见 [`docs/paper/RESTAD论文中文翻译.md`](docs/paper/RESTAD论文中文翻译.md)
- 改进点的完整实验框架见 [`thesis/`](thesis/)

> **想直接跑起来？** 直接看第四章「快速开始」。有三个坑必须先知道：
> 运行目录、数据集、`outputs/` 需手动建。

---

## 一、复现结果

加载仓库内 `restad/trained_models/` 的作者预训练权重做推理，逐项对照论文 Table 1 的
**RESTAD (R)** 行（论文只报两位小数）：

| 数据集 | RBF 中心数 | 指标 | 论文 | 本仓库复现 | 差值 |
|---|---|---|---|---|---|
| MSL | 128 | F1 / AUC-ROC / AUC-PR | 0.07 / 0.68 / 0.18 | 0.0674 / 0.6768 / 0.1837 | −0.0026 / −0.0032 / +0.0037 |
| PSM | 32 | 同上 | 0.15 / 0.79 / 0.59 | 0.1520 / 0.7879 / 0.5873 | +0.0020 / −0.0021 / −0.0027 |
| SMD | 256 | 同上 | 0.23 / 0.78 / 0.23 | 0.2264 / 0.7810 / 0.2250 | −0.0036 / +0.0010 / −0.0050 |

**9 项指标全部落在 ±0.005 以内**，即与论文两位小数的报告值完全一致。

复现脚本：`thesis/run_reproduction.py`（输出结果存档在 `thesis/results/*_reproduction/`）。

---

## 二、改进内容

所有改动**通过配置开关控制，默认值一律等于原作者行为**，因此不加任何 override 时，
本仓库的运行结果与原始实现完全一致（已做逐位回归验证）。

| 编号 | 层面 | 位置 | 改动 |
|---|---|---|---|
| **M1** | 模型结构 | `restad/Transformer_Model.py` | **门控残差注入**：原代码 `x = rbf_out` 让 RBF 输出整段顶替第二层编码器隐表示，主干语义被丢弃。改为 `x = W(x) + σ(g) ⊙ rbf_out`，主干走残差通路、RBF 走旁路，门控可学习 |
| **M2** | 训练目标 | `restad/RBF_Layer.py`、`restad/Training.py` | **RBF 参数正则**：中心两两余弦相似度惩罚（防中心坍缩）+ `log_gamma` 目标值回归（防核宽饱和） |
| **M3** | 评分口径 | `restad/Utils.py` | 重建误差可选 **L2 范数**（对齐论文式 (2) 的 ‖x−x̂‖₂，原代码实现的是 MSE）+ 归一化结果**裁剪到 [0,1]**，保证 `(1−εs) ∈ [0,1]` |
| **M4** | 实验协议 | `thesis/run_variants.py` | 原代码用**测试集损失**做早停、选最优、调学习率（数据泄漏）。新增 `--protocol val`：从训练序列尾部按时间切 20% 作验证集，测试集只在最后算分 |
| **M5** | 训练目标 | `restad/Training.py` | **密度对齐损失**：原训练目标只有重建 MSE，没有任何一项在推动「正常数据获得高 RBF 输出」。显式加入 `L_align = (z̄ − z_target)²` |

开关定义在 `restad/configs/model/Transformer_RBF.yaml`，全部默认为关闭状态。

---

## 三、实验结果

三数据集 × 多变体，**20 个随机种子**，评分采用「乘性融合 + 推理期时序 EMA 平滑」，
对同一种子下的 `Δ = 变体 − baseline` 做双侧配对 t 检验：

| 数据集 | 变体 | ΔAUC-ROC | p 值 |
|---|---|---|---|
| SMD | **M1** | **+0.0715** | **<0.001** |
| PSM | **M1** | **+0.0270** | **0.032** |
| MSL | **M1** | **+0.0141** | **0.031** |
| PSM | M1+M2+M5 | +0.0287 | 0.023 |
| SMD | M1+M2+M5 | +0.0687 | <0.001 |
| MSL | M3 | +0.0132 | 0.001 |

**机制诊断**（本工作的重要发现）：把 RBF 相似度**单独**当异常分数使用（完全不用重建误差）：

| 数据集 | baseline | M1+M2+M5 |
|---|---|---|
| MSL | 0.5045 | 0.5309 |
| PSM | 0.5719 | **0.6647** (p=0.001) |
| SMD | 0.5221 | **0.5911** (p=0.003) |

基线模型的相似度分支 AUC 仅 0.50~0.57，**接近随机猜测**。这说明原论文的检测提升主要来自
「乘性融合」这一评分形式，而非相似度分支本身携带的信息。M1 把相似度分支与主干特征解耦后，
再施加 M5 的显式优化目标，该分支才真正获得判别力。

**负结果也如实记录**：M2 单独使用无显著效果；M3 在 PSM 与 SMD 上显著为负。
这两项在论文中作为消融对照保留。

> 注意：p 值未做多重比较校正，主要结论依靠效应量与跨数据集的**一致性**支撑，而非单个 p 值。

---

## 四、快速开始

### 4.0 三条运行规则（最容易踩的坑）

| # | 规则 | 原因 |
|---|---|---|
| 1 | **`restad/main.py` 必须在 `restad/` 目录内运行** | 配置里是相对路径（`data_prefix: "datasets/MSL"`），在仓库根目录跑会 `FileNotFoundError` |
| 2 | **`outputs/<数据集>/` 需要手动创建** | 原代码直接往 `{save_prefix}/...` 写文件，**不会自动建目录** |
| 3 | `thesis/` 下的脚本会自行切换工作目录 | 这些脚本**在仓库根目录运行**即可，不需要 `cd` |

### 4.1 环境

```bash
conda create -n restad_env python=3.10
conda activate restad_env

# 先单独装 torch（版本见下表，不要交给 pip 自己挑）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 再装其余依赖
pip install -r thesis/requirements.txt
```

**关于 torch 与 numpy 的版本（不看会出问题）**

`thesis/requirements.txt` **故意不列 torch 和 numpy**。因为这两者版本必须配套，交给 pip 自动解析
会装出不兼容的组合 —— 典型症状是 torch 2.1 配了 numpy 2.x，`import torch` 直接抛
`_ARRAY_API not found` 之类的 ABI 错误。

已验证可跑通同一份代码的组合：

| 环境 | Python | torch | numpy | scikit-learn |
|---|---|---|---|---|
| 本地开发（本仓库实验数据的产出环境） | 3.10.20 | 2.5.1+cu121 | 2.2.6 | 1.7.2 |
| 云服务器（AutoDL PyTorch 镜像） | 3.10.8 | 2.1.2+cu121 | 1.26.3 | 1.3.2 |
| 原作者声明 | 3.8 | 2.2.2 | 1.24.4 | 1.3.2 |

三套环境都能跑通，但**浮点运算顺序不同，指标会有千分之几到百分之一的差异**，属正常现象。
本仓库 README 中的数字全部来自「本地开发」这一套。

显存需求很低：本工作实测在 **RTX 4050 Laptop（6 GB）** 上，显存占用约 1.3 GB。

### 4.2 数据集准备（仓库里没有，需单独获取）

仓库**不包含数据集**（共 306 MB，被 `.gitignore` 排除）。需要放置 9 个 `.npy` 文件：

```
restad/datasets/
├── MSL/   MSL_train.npy   MSL_test.npy   MSL_test_label.npy
├── PSM/   PSM_train.npy   PSM_test.npy   PSM_test_label.npy
└── SMD/   SMD_train.npy   SMD_test.npy   SMD_test_label.npy
```

| 数据集 | 维度 | 训练形状 | 测试形状 | 原始形态 |
|---|---|---|---|---|
| MSL | 55 | 58317 × 55 | 73729 × 55 | 官方 npy，可直接用 |
| PSM | 25 | 132481 × 25 | 87841 × 25 | **官方给的是 CSV，需转换** |
| SMD | 38 | 708405 × 38 | 708420 × 38 | 官方 npy，可直接用 |

**来源**：原论文指向 [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer)
的公开下载目录（Google Drive）。

**PSM 的转换方式**（官方未直接提供 npy，需要自己转）：

1. 读取 CSV，移除首列时间戳，保留 25 个特征列
2. 训练集有 4195 个 NaN、测试集没有；按上游 `PSM_loader` 的做法用 `np.nan_to_num` 填 0
3. 测试集特征列与标签的时间戳需逐项对齐
4. 训练集与测试集的特征列顺序必须一致

> 预处理的选择会影响复现数值，**论文中应当写明**：PSM 的 npy 是本地从 CSV 转换而来，
> 并非作者直接提供。

数据放置后，路径已配置好，无需改动 `restad/configs/dataset/*.yaml`：

```yaml
data_prefix:  "datasets/MSL"     # 相对于 restad/ 工作目录
save_prefix:  "outputs/MSL"
model_prefix: "trained_models"
```

### 4.3 验证跑通（先做这一步）

```bash
# 在仓库根目录运行；脚本会自动切换到 restad/ 并依次跑三个数据集
python thesis/run_reproduction.py
```

预期输出（与本仓库实测一致）：

```
数据集   指标        论文      复现        差值
MSL      F1-Score    0.07    0.0674   -0.0026
MSL      AUC-ROC     0.68    0.6768   -0.0032
MSL      AUC-PR      0.18    0.1837   +0.0037
PSM      F1-Score    0.15    0.1520   +0.0020
PSM      AUC-ROC     0.79    0.7879   -0.0021
PSM      AUC-PR      0.59    0.5873   -0.0027
SMD      F1-Score    0.23    0.2264   -0.0036
SMD      AUC-ROC     0.78    0.7810   +0.0010
SMD      AUC-PR      0.23    0.2250   -0.0050
```

差值都在 ±0.005 以内就说明复现成功。若某一项偏差明显偏大，先检查该数据集的
`rbf_dim` 是否正确（见第五章「常见问题」第 4 条）。

### 4.4 用原作者流程跑（可选）

```bash
cd restad
mkdir -p outputs/MSL outputs/PSM outputs/SMD    # 必须手动建，代码不会自动创建
python main.py --load_model True    # 加载作者权重推理（默认数据集为 MSL）
python main.py                      # 从零训练
```

切换数据集与模型配置：改 `restad/configs/base_config.yaml` 的 `defaults` 段，
或用 hydra override：`python main.py dataset=PSM model.rbf_dim=32`。

### 4.5 运行改进方案的消融实验

以下命令**在仓库根目录运行**：

```bash
# 单数据集、指定变体、单种子（M1 为例）
python thesis/run_variants.py --datasets PSM --variants baseline M1 --seeds 0 --protocol val

# 完整矩阵（多进程并发）
python thesis/run_matrix.py \
    --datasets PSM MSL SMD \
    --variants baseline M1 M3 M1+M2+M5 \
    --seeds 0 1 2 3 4 --workers 4 --protocol val

# 结果汇总：配对 t 检验 + Markdown 报告
python thesis/compare_variants.py --results <结果目录> --scoring mul_ema

# 出图
python thesis/make_figures.py --results <结果目录>

# 离线评分策略分析（不需要重训，直接读已存的 scores.npz）
python thesis/analyze_scores.py --results <结果目录>
```

`--protocol` 有两个取值：

- `author`：沿用原作者流程，用**测试集**做早停与模型选择（与论文对齐，但存在数据泄漏）
- `val`：从训练序列尾部按时间切 20% 作验证集，测试集不参与任何训练决策（本仓库主要结论用这个）

---

## 五、常见问题

**1. `FileNotFoundError: 'datasets/MSL/MSL_train.npy'`**

工作目录不对。`main.py` 必须在 `restad/` 下运行；`thesis/` 下的脚本在仓库根目录运行。

**2. 报错 `FileNotFoundError: 'outputs/MSL/...'` 或保存结果失败**

`outputs/` 目录不存在。原代码不会自动创建，先执行 `mkdir -p restad/outputs/{MSL,PSM,SMD}`。

**3. `import torch` 报 numpy 相关的 ABI 错误**

torch 与 numpy 版本不匹配（例如 torch 2.1 + numpy 2.x）。按 4.1 节的表格
重新装成配套版本。

**4. 加载作者权重时报 `size mismatch`**

`rbf_dim` 设错了。三个数据集的 RBF 中心数**各不相同**，必须按数据集设置：

| 数据集 | rbf_dim |
|---|---|
| MSL | 128 |
| PSM | 32 |
| SMD | 256 |

`thesis/run_reproduction.py` 已内置这张表，直接用它即可。

**5. 指标与论文差千分之几**

正常现象，原因见 4.1 节的版本说明（浮点运算顺序不同）。
论文只报两位小数，只要落在 ±0.005 内即视为复现成功。

**6. 数据集在 Google Drive 下载不下来**

可参考 [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) 仓库
issue 区中其他镜像来源，或从其他复现仓库获取同源数据（注意核对维度与形状，
见 4.2 节的表格）。

**7. 训练很慢 / 想跑完整实验矩阵**

本模型显存占用仅约 1.3 GB，**瓶颈是串行而非显存**。用 `thesis/run_matrix.py --workers N`
开多进程并发，同一张卡开 3~4 个进程即可显著提速。

---

## 六、目录结构

```
RESTAD/
├── README.md                  # 本文件
├── README_original.md         # 原作者 README 原文（署名）
├── LICENSE                    # 继承自上游仓库
├── Datasets_info.md           # 原作者的数据集说明
├── Hyperparameters_info.md    # 原作者的超参数说明
├── docs/paper/
│   └── RESTAD论文中文翻译.md    # 论文中文全文译稿
├── restad/                    # 模型源码（M1/M2/M3/M5 的改动都在这里）
│   ├── Transformer_Model.py   #   M1 门控残差注入
│   ├── RBF_Layer.py           #   M2 参数正则
│   ├── Training.py            #   M2 + M5 损失
│   ├── Utils.py               #   M3 评分口径
│   ├── configs/model/Transformer_RBF.yaml   # 所有改动开关
│   ├── datasets/              # ⚠️ 不在仓库中，需自行准备（见 4.2）
│   └── trained_models/        # 作者预训练权重（已包含）
└── thesis/                    # 实验框架（新增）
    ├── run_reproduction.py    # 复现论文数值 ← 先跑这个
    ├── run_variants.py        # 变体/种子/协议 扫描
    ├── run_matrix.py          # 多进程并发矩阵调度
    ├── analyze_scores.py      # 离线评分策略分析
    ├── compare_variants.py    # 配对统计比较
    ├── make_figures.py        # 论文图表生成
    ├── report_results.py      # 结果汇总
    ├── eval_reinfer.py        # 重叠窗口重推理
    ├── requirements.txt       # 依赖（不含 torch / numpy，原因见 4.1）
    ├── deploy/                # 云服务器部署脚本
    └── results/               # 各次实验的 metrics.json 与图表
```

---

## 七、引用与致谢

如果本仓库的改进对你的工作有帮助，请优先引用原论文：

```bibtex
@article{ghorbani2024restad,
  title={RESTAD: REconstruction and Similarity based Transformer for time series Anomaly Detection},
  author={Ghorbani, Ramin and Reinders, Marcel JT and Tax, David MJ},
  journal={arXiv preprint arXiv:2405.07509},
  year={2024}
}
```

原始实现版权归原作者所有，遵循上游仓库的 [LICENSE](LICENSE)。本仓库的改进部分
（`thesis/` 目录与 M1–M5 改动）为本科毕业设计成果。
