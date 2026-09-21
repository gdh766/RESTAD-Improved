# RESTAD 改进版：基于重建与相似度 Transformer 的时间序列异常检测

本科毕业设计成果。本仓库基于论文 **RESTAD: REconstruction and Similarity based Transformer for
time series Anomaly Detection**（Ghorbani, Reinders, Tax；arXiv:2405.07509）的
[官方实现](https://github.com/Raminghorbanii/RESTAD) 完成复现，并在此基础上做了 4 处改进。

- 原作者代码与说明见 [`README_original.md`](README_original.md)
- 论文中文译文见 [`docs/paper/RESTAD论文中文翻译.md`](docs/paper/RESTAD论文中文翻译.md)
- 改进点的完整实验框架见 [`thesis/`](thesis/)

---

## 一、复现结果

用作者提供的预训练权重直接推理，与论文表 1 逐项对照（论文只报两位小数）：

| 数据集 | 指标 | 论文 | 本仓库复现 |
|---|---|---|---|
| MSL | F1 / AUC-ROC / AUC-PR | 0.07 / 0.68 / 0.18 | 0.0674 / 0.6768 / 0.1837 |
| PSM | F1 / AUC-ROC / AUC-PR | 0.15 / 0.79 / 0.59 | 0.1361 / 0.7662 / 0.5604 |

复现脚本：`thesis/run_reproduction.py`。从零训练的链路也可正常运行（`python main.py`）。

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

开关定义在 `restad/configs/model/Transformer_RBF.yaml`。

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

### 环境

```bash
conda create -n restad_env python=3.10
conda activate restad_env
pip install -r thesis/requirements.txt      # 含原作者 requirements 的全部依赖
```

本工作实测环境：Python 3.10 / PyTorch 2.5.1+cu121 / RTX 4050 Laptop (6 GB)。

### 数据

数据集不在本仓库中，请按 [`Datasets_info.md`](Datasets_info.md) 提供的来源下载，
放到 `restad/datasets/` 下。相关的路径已在 `restad/configs/dataset/*.yaml` 中配置为相对路径。

### 复现论文数值

```bash
cd restad
python main.py --load_model True     # 用作者预训练权重推理
python main.py                        # 从零训练
```

### 运行改进方案的消融实验

```bash
# 单数据集、指定变体、单种子
python thesis/run_variants.py --datasets PSM --variants baseline M1 --seeds 0 --protocol val

# 完整矩阵（多进程并发 + 分片）
python thesis/run_matrix.py --datasets PSM MSL SMD \
    --variants baseline M1 M3 M1+M2+M5 --seeds 0 1 2 3 4 --workers 4 --protocol val

# 结果汇总与出图
python thesis/compare_variants.py --results <结果目录> --scoring mul_ema
python thesis/make_figures.py --results <结果目录>
```

---

## 五、目录结构

```
RESTAD/
├── README.md                  # 本文件
├── README_original.md         # 原作者 README 原文（署名）
├── LICENSE                    # 继承自上游仓库
├── Datasets_info.md           # 数据集来源说明
├── Hyperparameters_info.md    # 超参数说明
├── docs/paper/
│   └── RESTAD论文中文翻译.md    # 论文中文全文译稿
├── restad/                    # 模型源码（M1/M2/M3/M5 的改动都在这里）
│   ├── Transformer_Model.py   #   M1 门控残差注入
│   ├── RBF_Layer.py           #   M2 参数正则
│   ├── Training.py            #   M2 + M5 损失
│   ├── Utils.py               #   M3 评分口径
│   └── configs/model/Transformer_RBF.yaml   # 所有改动开关
└── thesis/                    # 实验框架（新增）
    ├── run_reproduction.py    # 复现论文数值
    ├── run_variants.py        # 变体/种子/协议 扫描
    ├── run_matrix.py          # 多进程并发矩阵调度
    ├── analyze_scores.py      # 离线评分策略分析
    ├── compare_variants.py    # 配对统计比较
    ├── make_figures.py        # 论文图表生成
    └── results/               # 逐次实验的 metrics.json
```

---

## 六、引用与致谢

如果本仓库的改进对你的工作有帮助，请优先引用原论文：

```bibtex
@article{ghorbani2024restad,
  title={RESTAD: REconstruction and Similarity based Transformer for time series Anomaly Detection},
  author={Ghorbani, Ramin and Reinders, Marcel JT and Tax, David MJ},
  journal={arXiv preprint arXiv:2405.07509},
  year={2024}
}
```

原始实现版权归原作者所有，遵循上游仓库的 [LICENSE](LICENSE)。
