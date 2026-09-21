# thesis/ —— 毕设实验代码

这个目录放的是**我们自己写的**实验与评测代码。`RESTAD/restad/` 下的原作者代码
只在必要处加了开关（见下），默认值全部等于原论文行为。

## 目录内容

| 文件 | 作用 |
|---|---|
| `run_variants.py` | 消融主运行器：训练 + 评分，一次落盘配置/指标/逐点分数/权重 |
| `analyze_scores.py` | 离线评分策略分析：在已存的逐点分数上试各种融合与平滑方式，不用重训 |
| `eval_reinfer.py` | 用已存权重做重叠窗口重推理，验证"推理期改动" |
| `compare_variants.py` | 合并多个结果目录，做同种子配对统计，输出论文结果表 |
| `results/` | 每次运行一个时间戳目录，互不覆盖 |

## 环境

```powershell
# 工作区根目录下
.\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\run_variants.py --help
```

环境由工作区根目录的 `.venv-restad` 提供（`--system-site-packages` 方式复用本机 PyTorch）。

## 复现一条完整链路

```powershell
# 1) 训练 + 评分（严格协议：从训练序列尾部切 20% 作验证集）
.\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\run_variants.py `
    --datasets MSL PSM --variants baseline M1 M3 "M1+M2+M5" `
    --seeds 0 1 2 3 4 --protocol val

# 2) 配对统计，输出论文用结果表
.\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\compare_variants.py `
    --results RESTAD\\thesis\\results\\<时间戳>_val --scoring mul_ema
```

## 数据与协议上的几个必须写进论文的点

1. **窗口不重叠**：原作者用 `step == window_size == 100`。序列尾部不足一窗的点被丢弃
   （MSL 测试集 73729 个点里最后 29 个没参与评价）。
2. **阈值口径**：`find_threshold` 把训练分数与测试分数拼在一起取分位数，
   属于用到测试分布的离线协议，不等于部署时能预先确定阈值。
3. **点调整**：原代码的 AUC-PR 实际调用 `average_precision_score`（AP），
   且全程不使用 point adjustment。
4. **两种协议**：
   - `--protocol author`：原作者流程，模型选择/早停/学习率调度都看**测试集**损失。
     保留它只是为了和论文表格对齐，它不是严格的独立测试。
   - `--protocol val`：从训练序列**尾部按时间顺序**切 20% 作验证集，
     测试集只在最后评分时出现一次。论文的主表用这个协议。
     代价是验证集只覆盖训练序列末段的数据分布。
5. **交叉验证**：本项目未做按机器/按实体的独立划分，不能据此声称跨实体泛化。

## 三处修改与开关

| 编号 | 位置 | 开关（默认都是关闭） |
|---|---|---|
| M1 | `restad/Transformer_Model.py` `Encoder` | `model.use_residual_rbf` |
| M2 | `restad/RBF_Layer.py` `regularization()` + `restad/Training.py` | `model.center_div_lambda`、`model.gamma_reg_lambda`、`model.gamma_reg_target` |
| M3 | `restad/Utils.py` `calculate_reconstruction_errors` / `minmax_fit_transform` | `model.rec_error_type`、`model.robust_norm` |
| M5 | `restad/Training.py` | `model.density_align_lambda`、`model.density_align_target` |

验证开关默认关闭时行为与原作者一致：跑 `restad/main.py` 得到的 MSL 指标
（F1 0.0666 / AUC 0.6350 / AP 0.1712）在加开关前后逐位相同。
