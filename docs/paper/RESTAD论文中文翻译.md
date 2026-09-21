# RESTAD：面向时间序列异常检测的「重建 + 相似度」Transformer

**原文标题**：RESTAD: REconstruction and Similarity based Transformer for time series Anomaly Detection
**作者**：Ramin Ghorbani¹\*，Marcel J.T. Reinders¹，David M.J. Tax¹
**单位**：¹ 代尔夫特理工大学（Delft University of Technology）模式识别实验室，荷兰代尔夫特
**通讯作者**：r.ghorbani@tudelft.nl
**来源**：arXiv:2405.07509v1 [cs.LG]，2024-05-13（预印本）
**代码仓库**：https://github.com/Raminghorbanii/RESTAD

> 译者注：本文为毕业论文研读用的非官方中译本，按原文结构逐段对应翻译。公式、表格数值、图注均保留原样。专有名词首次出现时以「中文（English）」形式给出，此后沿用中文；`RBF`、`Transformer`、`AUC` 等已在中文文献中通行的缩写不再译出。

---

## 摘要

时间序列异常检测在众多领域中至关重要。此类任务标注数据稀缺，使得无监督学习方法受到越来越多的关注。这类方法往往仅依赖重建误差，因而通常难以在复杂数据集中检测出细微异常。为解决该问题，我们提出 **RESTAD**：在 Transformer 架构内部嵌入一层**径向基函数（Radial Basis Function, RBF）神经元**，从而得到 Transformer 的一个改型。该 RBF 层在隐表示上拟合一个非参数密度，因此 RBF 输出值高即表示当前数据点与训练集中占主导的正常数据相似。RESTAD 将 RBF 相似度分数与重建误差融合，以提高对异常的敏感度。我们的实证评估表明，RESTAD 在多个基准数据集上优于多种已有基线。

**关键词**：时间序列，异常检测，径向基函数（RBF）核，Transformer

---

## 1 引言

时间序列中的异常——即偏离正常行为的、意料之外的模式——在从金融欺诈到危及生命的健康问题等各个领域都可能预示着严重问题。因此，准确的异常检测十分重要。鉴于异常本身稀少、因而缺少足够的标注数据，完全监督的方法并不适用。相应地，无监督学习方法获得了越来越多的关注[1]。这类方法并不显式要求提供标注的异常样本，因而非常适合检测未知的或意料之外的异常[2]。

距离型单类支持向量机（One-Class SVM, OC-SVM）[3]与密度型局部离群因子（Local Outlier Factor, LOF）[4]等经典无监督技术已被广泛使用。然而，它们难以应对时间序列数据的时间依赖、高维性以及复杂的泛化需求[5]。深度学习领域的新进展为解决这些挑战提供了有前景的方案[6]。Transformer 与 LSTM 等架构善于捕捉时间模式，并能从时间序列数据中自动学习分层且非线性的特征[7, 8, 9]。

在上述进展的基础上，研究者提出了多种有效的异常检测方法，它们大多以重建误差作为主要的异常判据[10, 11, 12]。这类方法通常通过衡量输入与其重建结果之间的偏差来识别异常。其基本假设是：典型数据的重建误差较低，而异常数据由于模型对这类模式不熟悉，会表现出更高的误差[11, 13, 14]。

使用重建误差做异常检测的一个主要问题是**过度泛化**[15]。以拟合训练数据中主导模式为目标的模型，会把这种拟合能力泛化到细微的变异上。因此，细微异常也可能被这些模型很好地重建出来。结果是，这些异常与典型模式之间的区分度下降，模型的检测敏感度随之降低[16]。这一效应如图 1(a) 所示：原始信号在时刻 $t_0$ 包含一个细微异常，在 $t_1$ 包含一个显著异常。重建信号是原始信号的轻微平滑版本，若仅使用重建误差，细微异常会被漏掉——因为其重建误差仍低于检测阈值，如图 1(b) 所示。

为改进无监督异常检测，已有工作在传统的基于重建误差的异常分数之外，引入了其他类型的分数。例如，AnomalyTrans[17] 利用**关联差异（association discrepancy）**的概念，即考虑某一时间点与其相邻时间点的相似性，据此对重建误差重新加权，从而构造最终的复合异常分数。然而，该方法中包含一项归一化操作，在不存在异常的时段，这项归一化会放大正常时间点的差异分数，可能造成误报。这会把正常数据点误导性地凸显为异常。虽然该做法对识别明显的离群点有效，但它也可能在无意中把正常的细小波动错误地表征为异常。

为克服基于重建误差评分所带来的困难，以及关联差异方法的局限，我们提出将重建误差与一种专门的非线性变换——径向基函数（RBF）核[18]——相结合。RBF 核产生一个相似度分数，用以衡量数据点与某个参考点（或中心）的接近程度，这使其在异常检测中极为有效。异常数据点偏离（远离）典型模式，与 RBF 核产生的相似度分数更低，因而该分数可直接度量一个点的异常程度。这一分数能有效补充重建误差，并提升对可能被重建误差忽略的细微异常的敏感度。图 1(c) 展示了将 RBF 分数与重建误差结合的有效性：RBF 核作用于 Transformer 隐表示中的典型数据。通过把重建误差与 RBF 相似度分数结合，我们得到了一种综合性的复合异常分数，它既捕捉了相对预期模式的偏离，又能保证细微异常仍被标记出来。该复合异常分数见图 1(d)，两类异常的异常分数此时均已高于检测阈值。

本文提出了 Transformer 模型的一种改型。选择 Transformer，是因为它具备捕捉序列数据中时间依赖的能力。通过把 RBF 神经元融入 Transformer 架构，我们得到一个能够协同利用相似度分数与重建误差、进而计算独特异常分数的模型。通过大量评估，我们证明这一新的「面向时间序列异常检测的、基于重建与相似度的 Transformer」——RESTAD——在一系列基准数据集上优于现有基线。

---

## 2 方法

假设观测到的时间序列数据集由 $N$ 条长度为 $T$ 的序列组成。该数据集中的每一条序列记为 $X_i = \{x_{i,t}\}_{t=1}^{T}$，其中 $x_{i,t}$ 表示第 $i$ 条序列在时刻 $t$ 的观测值，具有 $d$ 个维度，即 $x_{i,t} \in \mathbb{R}^{d}$。我们的任务是判定给定的 $x_{i,t}$ 是否表现出异常行为。

### 2.1 RESTAD 框架

在本研究中，我们通过一层专门的 RBF 神经元，把异常检测机制嵌入到原始 Transformer[8] 中，见图 1(c)。该 RBF 层作用于来自前一层的隐表示，记为 $H_i = \{h_{i,t}\}_{t=1}^{T}$，其中 $h_{i,t} \in \mathbb{R}^{d_h}$。该层计算每个数据点 $h_{i,t}$ 与一组可学习的参考点（即中心）$C = \{c_m\}_{m=1}^{M}$ 的相似度，其中 $c_m \in \mathbb{R}^{d_h}$。该计算得到 RBF 输出 $Z_i = \{z_{i,t}\}_{t=1}^{T}$，其中 $z_{i,t} \in \mathbb{R}^{M}$，它随后作为模型后续层的输入。具体而言，每个数据点相对于每个中心的 RBF 相似度输出定义为：

$$
z^{m}_{i,t}(h_{i,t}, c_m) = \exp\!\left(-\frac{1}{2}\, e^{\gamma}\, \lVert h_{i,t} - c_m \rVert^{2}\right) \tag{1}
$$

这里参数 $\gamma$ 控制 RBF 的宽度，影响它如何对待与中心距离不同的数据点。该参数在训练中被初始化并调整。对 $\gamma$ 取指数可保证尺度参数为正，从而在不额外施加正性约束的前提下简化优化过程。

**异常分数**：RESTAD 通过最小化均方误差（Mean Squared Error, MSE）来训练，以实现准确的重建。在异常检测阶段，引入一个复合异常分数 $\mathrm{RESTAD}_{score}$，由归一化后的 RBF 相似度分数与重建误差组合而成。归一化采用 MinMax，以保证两者可比。RBF 相似度分数衡量 $x_{i,t}$ 与已学习中心的吻合程度：相似度越高，说明行为越正常；相似度越低（即距 RBF 中心的距离越大），则表明异常。该分数通过对所有中心上的 RBF 输出 $z_{i,t}$ 取平均得到。重建误差是真实数据 $x_{i,t}$ 与其重建结果 $\hat{x}_{i,t}$ 之间的平方差。$\mathrm{RESTAD}_{score}$ 的形式为：

$$
\mathrm{RESTAD}_{score}(x_{i,t}) = \epsilon_r \times \epsilon_s \tag{2}
$$

其中 $\epsilon_r = \lVert x_{i,t} - \hat{x}_{i,t} \rVert_2$ 表示重建误差，$\epsilon_s = \left(1 - \frac{1}{M}\sum_{m=1}^{M} z^{m}_{i,t}\right)$ 度量不相似度。这一组合既能凸显那类「重建误差低、RBF 分数也低」的细微异常，也能凸显「重建误差高或 RBF 分数低」的显著异常。

**RBF 层参数的初始化**：RBF 参数（包括中心 $c$ 与尺度参数 $\gamma$）的恰当初始化对本方法至关重要。我们考察了两种初始化策略——随机初始化与 K-means 初始化——以评估它们对模型性能的影响。随机初始化时，参数 $c$ 与 $\gamma$ 从均值为零、标准差为一的正态分布中采样。该方法虽然简单，但可能导致收敛较慢、陷入局部极小值的风险，并且初始阶段可能无法有效代表数据分布，进而造成不稳定。相比之下，K-means 初始化利用数据自身的内在结构，获得更具代表性的起点。该做法先训练一个基础模型（不含集成的 RBF 层）以最小化重建的 MSE：

$$
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}\left\lVert X_i - \hat{X}_i \right\rVert_F^{2} \tag{3}
$$

当基础模型达到满意的重建精度后，从 RBF 层计划嵌入的那一层提取隐表示。随后用该表示通过 K-means 聚类算法初始化 $c$。尺度参数 $\gamma$ 用 $\tilde{\sigma}^{2}$ 初始化，后者是每个数据点到其最近聚类中心的平方距离的均值：

$$
\tilde{\sigma}^{2} = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T}\min_{m}\left\lVert h_{i,t} - c_m \right\rVert^{2},\quad \forall m \in [1, M] \tag{4}
$$

这里 $h_{i,t}$ 表示第 $i$ 个样本在第 $t$ 个时间步的隐表示向量，$c_m$ 是 K-means 算法得到的第 $m$ 个聚类中心。该值 $\tilde{\sigma}^{2}$ 用于将 $\gamma$ 初始化为 $\gamma = \frac{1}{\tilde{\sigma}^{2}}$，从而保证 RBF 函数的展宽由数据点围绕各自中心的平均离散程度所决定。

---

## 3 实验设置

### 3.1 数据集与预处理

我们在实验中使用了三个公开且被广泛采用的基准数据集：1) 服务器机器数据集（Server Machine Dataset, SMD）[11]；2) 火星科学实验室（Mars Science Laboratory, MSL）巡视器数据集[9]；3) 池化服务器指标数据集（Pooled Server Metrics, PSM）[19]。各数据集的更多信息见我们的代码仓库¹。

数据预处理包括：对每个特征在时间维度上归一化为零均值、单位方差。随后，将归一化后的信号切分为**不重叠**的滑动窗口[20]，窗口长度固定为 100 个数据点，这是基于此前相关工作[17, 10]的常用设置。

### 3.2 实现细节

**RESTAD 模型**：RESTAD 是原始 Transformer 的一种改型，按图 2 所示嵌入了 RBF 核层。它包含一个数据嵌入（DataEmbedding）模块，同时结合了 token 嵌入与位置嵌入；其后是三层编码器。每一层都包含多头自注意力机制与前馈网络。模型的隐维度为 32，前馈网络的中间层维度为 128，注意力头数为 8。RBF 层置于第二个编码器层之后（其他放置位置同样可行，见第 4.1 节）。优化使用 Adam 优化器，超参数通过对重建任务性能的系统搜索确定。更多超参数细节见我们的代码仓库¹。

**评价**：超过阈值 $\delta$ 的异常分数（式 2）被判定为异常。针对依赖阈值的评价，采用 F1 分数。这里我们沿袭[17]的做法，将 $\delta$ 设定为把预先给定比例的数据点标记为异常（SMD 为 0.5%，其他数据集为 1%）。针对与阈值无关的分析，我们使用 AUC-ROC、AUC-PR、VUS-ROC 与 VUS-PR 指标[21]。由于**点调整（point adjustment）**方法[22]会高估性能[23]，我们不使用它。我们的模型与以下基线和当前最优模型进行比较：LSTM[9]、原始 Transformer[17]、USAD[13]、PatchAD[12]、AnomalyTrans[17] 与 DCdetector[10]。

> ¹ https://github.com/Raminghorbanii/RESTAD

---

## 4 结果

如表 1 所示，我们的实证结果凸显了 RESTAD 在异常检测上的有效性。无论采用哪种 RBF 初始化策略，RESTAD 在所有基准数据集与全部评价指标上都优于所有基线模型。两种初始化方法之间存在轻微的性能差异，但这些差异尚不足以确立其中一种方法优于另一种。

为直观展示检测差异，图 3 给出 SMD 数据集某一短片段上各模型的异常分数。PatchAD、DCdetector 与 AnomalyTrans 模型出现大量误检：DCdetector 表现出重复误报的模式，PatchAD 的评分则近乎随机。LSTM、USAD 与 Transformer 模型要么漏检部分异常，要么检测得很弱；例如第一段异常区域未被 USAD 检出，而 LSTM 与 Transformer 也仅微弱检出。相比之下，RESTAD 模型展现出稳健的检测能力，有效识别出了全部异常区段。

**表 1：基线与 RESTAD 在测试集上的性能指标。** 初始化方法以 (R) 表示随机初始化，(K) 表示 K-means 初始化。所有指标均为数值越高表示异常检测性能越好。

| 数据集 | | SMD | | | | | MSL | | | | | PSM | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **模型** | | F1 | AUC-ROC | AUC-PR | VUS-ROC | VUS-PR | F1 | AUC-ROC | AUC-PR | VUS-ROC | VUS-PR | F1 | AUC-ROC | AUC-PR | VUS-ROC | VUS-PR |
| LSTM | | 0.12 | 0.74 | 0.17 | 0.79 | 0.20 | 0.06 | 0.56 | 0.14 | 0.63 | 0.19 | 0.11 | 0.73 | 0.50 | 0.72 | 0.51 |
| USAD | | 0.13 | 0.63 | 0.11 | 0.72 | 0.14 | 0.06 | 0.53 | 0.14 | 0.59 | 0.18 | 0.07 | 0.60 | 0.41 | 0.61 | 0.43 |
| PatchAD | | 0.01 | 0.50 | 0.04 | 0.61 | 0.08 | 0.03 | 0.50 | 0.10 | 0.57 | 0.15 | 0.02 | 0.50 | 0.28 | 0.55 | 0.33 |
| Transformer | | 0.11 | 0.75 | 0.19 | 0.80 | 0.22 | 0.06 | 0.56 | 0.14 | 0.63 | 0.19 | 0.13 | 0.71 | 0.49 | 0.70 | 0.50 |
| AnomalyTrans | | 0.03 | 0.49 | 0.04 | 0.50 | 0.07 | 0.02 | 0.49 | 0.10 | 0.52 | 0.14 | 0.02 | 0.51 | 0.30 | 0.53 | 0.34 |
| DCDetector | | 0.01 | 0.50 | 0.04 | 0.51 | 0.08 | 0.02 | 0.50 | 0.11 | 0.58 | 0.15 | 0.02 | 0.50 | 0.28 | 0.52 | 0.32 |
| **RESTAD (R)** | | **0.23** | **0.78** | **0.23** | **0.82** | **0.24** | **0.07** | **0.68** | **0.18** | **0.72** | **0.23** | **0.15** | **0.79** | **0.59** | **0.76** | **0.57** |
| RESTAD (K) | | 0.20 | 0.79 | 0.24 | 0.83 | 0.25 | 0.07 | 0.66 | 0.18 | 0.71 | 0.23 | 0.14 | 0.79 | 0.57 | 0.76 | 0.56 |

### 4.1 消融分析

消融实验基于随机初始化的 RBF 层。这一选择依据是我们的发现：随机初始化与 K-means 策略同样有效（见表 1），同时更为简单、计算效率更高。

**异常分数判据**：表 2 凸显了将 RBF 分数整合进异常检测所带来的影响。将 RBF 层的不相似度分数（$\epsilon_s$）与重建误差（$\epsilon_r$）相乘构成复合异常分数（$\epsilon_s \times \epsilon_r$），被证明最为有效，在所有基准与所有指标上均持续带来提升。在原始 Transformer 上加入 RBF 层、但仅以重建误差 $\epsilon_r$ 作为异常分数，只在部分数据集上带来边际改善。相比之下，仅使用不相似度分数（$\epsilon_s$），或把它直接加到重建误差上（$\epsilon_s + \epsilon_r$），都没有表现出显著收益。

图 4 通过展示三个数据集各自的片段及其对应的异常分数，直观说明了我们的复合异常分数相对于传统重建分数（$\epsilon_r$）的优越性。对于仅依赖重建误差的模型所忽略的异常，我们的异常分数能够有效识别，且检测响应明显更强，通常超过阈值。需要注意的是，图中所示的阈值是基于整个数据集优化得到的最优阈值。若针对图中所示的数据子集去调整该阈值，会削弱整体性能，因此这种做法并不可行。

**RBF 层的放置位置**：我们通过在三个编码器层中每一层之后分别嵌入 RBF 层，考察了 RBF 层放置位置的灵活性。图 5 表明，无论 RBF 层位于何处，各数据集上的性能都保持稳健。需要注意的是，将 RBF 层放在第二个编码器层之后，在所有数据集上都取得了略优的性能。这一微小优势影响了我们的决策，最终模型架构（见图 2）把 RBF 层放在第二层之后。

**表 2：整合 RBF 层及异常分数选择的影响。** 所有指标均为数值越高表示异常检测性能越好。

| 架构 | 异常判据 | SMD F1 | SMD ROC | SMD PR | SMD VUS-ROC | SMD VUS-PR | MSL F1 | MSL ROC | MSL PR | MSL VUS-ROC | MSL VUS-PR | PSM F1 | PSM ROC | PSM PR | PSM VUS-ROC | PSM VUS-PR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | $\epsilon_r$ | 0.11 | 0.75 | 0.19 | 0.80 | 0.22 | 0.06 | 0.56 | 0.14 | 0.63 | 0.19 | 0.13 | 0.71 | 0.49 | 0.70 | 0.50 |
| RESTAD | $\epsilon_r$ | 0.11 | 0.77 | 0.18 | 0.81 | 0.21 | 0.07 | 0.63 | 0.16 | 0.69 | 0.21 | 0.13 | 0.75 | 0.56 | 0.74 | 0.55 |
| RESTAD | $\epsilon_s$ | 0.01 | 0.44 | 0.03 | 0.52 | 0.07 | 0.01 | 0.43 | 0.08 | 0.48 | 0.12 | 0.01 | 0.32 | 0.20 | 0.37 | 0.25 |
| RESTAD | $\epsilon_r + \epsilon_s$ | 0.04 | 0.57 | 0.06 | 0.60 | 0.10 | 0.07 | 0.61 | 0.16 | 0.65 | 0.20 | 0.01 | 0.68 | 0.49 | 0.59 | 0.45 |
| RESTAD | $\epsilon_r \times \epsilon_s$ | **0.23** | **0.78** | **0.23** | **0.82** | **0.24** | **0.07** | **0.68** | **0.18** | **0.72** | **0.23** | **0.15** | **0.79** | **0.59** | **0.76** | **0.57** |

**RBF 中心的数量**：图 6 展示了 RESTAD 中 RBF 层中心数量从 8 到 512 变化所带来的影响。结果表明，最优的 RBF 中心数量依赖于具体数据集。此外，超过某一阈值之后，继续增加中心数量不仅不会提升性能，反而可能使性能下降。

---

## 5 讨论与结论

我们提出了 RESTAD——一种面向无监督异常检测的 Transformer 改型，它改进了仅以重建误差作为异常分数的局限。通过将 RBF 层整合进 Transformer，我们把 RBF 相似度分数与重建误差结合起来，提升了对细微异常的敏感度。RESTAD 在多个数据集与多种评价指标上持续优于已有基线。

我们的发现表明：RESTAD 的性能对 RBF 层的初始化方法相对不敏感，说明其对初始化差异具有稳健性。性能的显著提升主要归因于 RBF 相似度分数与重建误差的**乘性融合**，这明显改善了异常检测能力。RBF 层在架构中的放置位置对性能没有显著影响，说明在整合 RBF 层时架构上具有灵活性。然而，最优的 RBF 中心数量依赖具体数据集。这些发现为未来研究提供了动机：将 RBF 层整合到其他深度学习架构中以服务异常检测任务。

---

## 致谢

**资助**：本工作由荷兰研究理事会（Dutch Research Council, NWO）资助，项目编号 628.011.214。

---

## 参考文献

[1] Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. *ACM computing surveys (CSUR)*, 41(3):1–58, 2009.

[2] Ramin Ghorbani, Marcel JT Reinders, and David MJ Tax. Personalized anomaly detection in ppg data using representation learning and biometric identification. *Biomedical Signal Processing and Control*, 94:106216, 2024.

[3] Bernhard Schölkopf, Robert C Williamson, Alex Smola, John Shawe-Taylor, and John Platt. Support vector method for novelty detection. *Advances in neural information processing systems*, 12, 1999.

[4] Markus M Breunig, Hans-Peter Kriegel, Raymond T Ng, and Jörg Sander. Lof: identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD international conference on Management of data*, pages 93–104, 2000.

[5] Nesryne Mejri, Laura Lopez-Fuentes, Kankana Roy, Pavel Chernakov, Enjie Ghorbel, and Djamila Aouada. Unsupervised anomaly detection in time-series: An extensive evaluation and analysis of state-of-the-art methods. *arXiv preprint arXiv:2212.03637*, 2022.

[6] Kukjin Choi, Jihun Yi, Changhwa Park, and Sungroh Yoon. Deep learning for anomaly detection in time-series data: review, analysis, and guidelines. *IEEE Access*, 9:120043–120065, 2021.

[7] Shreshth Tuli, Giuliano Casale, and Nicholas R Jennings. Tranad: Deep transformer networks for anomaly detection in multivariate time series data. *arXiv preprint arXiv:2201.07284*, 2022.

[8] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[9] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 387–395, 2018.

[10] Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun. Dcdetector: Dual attention contrastive representation learning for time series anomaly detection. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 3033–3045, 2023.

[11] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 2828–2837, 2019.

[12] Zhijie Zhong, Zhiwen Yu, Yiyuan Yang, Weizheng Wang, and Kaixiang Yang. Patchad: Patch-based mlp-mixer for time series anomaly detection. *arXiv preprint arXiv:2401.09793*, 2024.

[13] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A Zuluaga. Usad: Unsupervised anomaly detection on multivariate time series. In *Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 3395–3404, 2020.

[14] Daehyung Park, Yuuna Hoshi, and Charles C Kemp. A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational autoencoder. *IEEE Robotics and Automation Letters*, 3(3):1544–1551, 2018.

[15] Tianzi Zhao, Liang Jin, Xiaofeng Zhou, Shuai Li, Shurui Liu, and Jiang Zhu. Unsupervised anomaly detection approach based on adversarial memory autoencoders for multivariate time series. *Computers, Materials & Continua*, 76(1), 2023.

[16] Haoyi Zhong, Yongjiang Zhao, and Chang Gyoon Lim. Abnormal state detection using memory-augmented autoencoder technique in frequency-time domain. *KSII Transactions on Internet and Information Systems (TIIS)*, 18(2):348–369, 2024.

[17] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy. *arXiv preprint arXiv:2110.02642*, 2021.

[18] Mark JL Orr et al. Introduction to radial basis function networks, 1996.

[19] Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. Practical approach to asynchronous multivariate time series anomaly detection and localization. In *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining*, pages 2485–2494, 2021.

[20] Lifeng Shen, Zhuocong Li, and James Kwok. Timeseries anomaly detection using temporal hierarchical one-class network. *Advances in Neural Information Processing Systems*, 33:13016–13026, 2020.

[21] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S Tsay, Aaron Elmore, and Michael J Franklin. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. *Proceedings of the VLDB Endowment*, 15(11):2774–2787, 2022.

[22] Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, et al. Unsupervised anomaly detection via variational autoencoder for seasonal kpis in web applications. In *Proceedings of the 2018 world wide web conference*, pages 187–196, 2018.

[23] Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon. Towards a rigorous evaluation of time-series anomaly detection. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 7194–7201, 2022.

---

## 图注（原文图 1–6）

**图 1**：传统重建式异常检测与 RBF 增强式异常检测的对比。
(a) 原始信号（含细微异常与显著异常）及其重建信号。
(b) (a) 中信号的重建误差，凸显了检测细微异常时的困难。
(c) 整合 RBF 后模型的可视化：二维散点图展示了典型数据、细微异常、显著异常，以及 RBF 中心及其影响半径，体现 RBF 区分典型点与异常点的能力。
(d) 使用 RBF 后的增强异常分数，显示出对细微异常检测能力的改善。

**图 2**：所提 RESTAD 模型概览。RBF 层被添加在第二个编码器层之后。

**图 3**：不同模型在 SMD 数据集某一片段上的异常分数。红色高亮区域表示真实的异常时段（由专家标注）。

**图 4**：在各数据集的片段上，我们的复合异常分数（$\epsilon_r \times \epsilon_s$）与重建误差（$\epsilon_r$）的效果对比。红色高亮区域表示真实的异常时段（由专家标注）。

**图 5**：RESTAD 在不同 RBF 层放置位置下的性能。

**图 6**：RESTAD 在 RBF 中心数量变化下的平均性能。阴影区域表示 ±1 标准差，用以体现多次运行间的波动。

---

## 附录：译名对照

| 英文 | 中文 | 说明 |
|---|---|---|
| Reconstruction error ($\epsilon_r$) | 重建误差 | 输入与其重建结果之差的 L2 范数 |
| Dissimilarity score ($\epsilon_s$) | 不相似度分数 | $1 - \text{mean}(z)$，RBF 输出均值取补 |
| Composite anomaly score | 复合异常分数 | $\epsilon_r \times \epsilon_s$ |
| Radial Basis Function (RBF) | 径向基函数 | 本文核心组件 |
| Association discrepancy | 关联差异 | AnomalyTrans 提出的判据 |
| Over-generalization | 过度泛化 | 重建式方法的核心弊端 |
| Point adjustment | 点调整 | 会高估性能，本文不使用 |
| Volume Under the Surface (VUS) | 曲面下体积 | 对阈值不敏感的评价指标 |
| Window / non-overlapped sliding window | 窗口 / 不重叠滑动窗口 | 长度 100 的切分方式 |
| Initialization: Random / K-means | 初始化：随机 / K-means | 两种 RBF 参数初始化策略 |
