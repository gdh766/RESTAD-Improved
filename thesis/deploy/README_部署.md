# RESTAD 毕设实验：服务器部署说明

## 一、机器怎么选

| 项 | 建议 | 原因 |
|---|---|---|
| 显卡 | **单卡，显存 ≥ 16 GB** | 每个训练进程约占 1.3~1.8 GB，8 并发需要 16 GB 以上 |
| 镜像 | PyTorch 2.x + CUDA 11.8/12.1（AutoDL 的 `PyTorch 2.x` 基础镜像即可） | 镜像自带 torch，省掉几个 GB 下载 |
| CPU | 4 核以上 | 数据加载是次要瓶颈，核数太少会拖慢 |
| 系统盘 | ≥ 10 GB 空闲 | 代码 + 数据集约 1.5 GB，结果约 1 GB |

显存对照：**16 GB 开 8 并发；12 GB 开 6；8 GB 开 4；6 GB 开 3**。

预计总耗时（420 组，8 并发）：**约 1~1.5 小时**。
其中 SMD 占大头（单次约 4 分钟），MSL 约 20 秒，PSM 约 35 秒。

## 二、把包传上去

本地已经打好包：`restad_thesis_deploy.tar.gz`（约 121 MB，含三个数据集的 `.npy`，
所以服务器上**不用再下载任何数据**）。

两种传法，任选一种：

**A. 命令行（本机 PowerShell 执行）**

```powershell
scp -P <端口> restad_thesis_deploy.tar.gz root@<服务器IP>:/root/
```

**B. 平台网页上传**

AutoDL 等平台的 JupyterLab 左侧文件树里直接拖进去即可。

然后在服务器上解压：

```bash
cd /root
tar xzf restad_thesis_deploy.tar.gz
cd restad_thesis
ls                                     # 应看到 RESTAD/ 这个目录
```

> 用的是 tar.gz 而不是 zip：Windows 上打 zip 会把路径写成反斜杠，Linux 解压出来
> 会变成一堆名字里带 `\` 的怪文件。tar.gz 没有这个问题，且服务器上一定有 tar。

## 三、跑起来（两条命令）

```bash
cd /root/restad_thesis

# 1) 装依赖 + 自检（几十秒到几分钟，取决于镜像里有没有预装 torch）
bash RESTAD/thesis/deploy/setup_env.sh

# 2) 正式实验：3 数据集 × 7 变体 × 20 种子 = 420 组，8 并发
bash RESTAD/thesis/deploy/run_matrix.sh
```

不放心的话，中间可以先插一条 2 分钟的冒烟检查：

```bash
bash RESTAD/thesis/deploy/smoke.sh
```

想改规模就设这几个环境变量：

```bash
# 只跑 MSL 和 PSM，16 个种子，6 并发
WORKERS=6 DATASETS="MSL PSM" SEEDS="$(seq 0 15)" bash RESTAD/thesis/deploy/run_matrix.sh
```

## 四、跑的过程中怎么看

```bash
# 总共完成多少组
ls -d RESTAD/thesis/results/*_full/*/ | wc -l

# 实时看各分片在干什么
tail -f RESTAD/thesis/results/*_full/shard0.log

# 看显卡占用（确认并发确实跑满了）
nvidia-smi
```

每个 run 目录里都有：`config.yaml`（解析后的完整配置）、`metrics.json`（指标）、
`scores.npz`（逐点分数，后面画图和对齐都用它）、`model.pth`（权重）。

## 五、跑完把结果带回来

脚本跑完会自动做三件事：配对统计、生成图表、提示打包命令。回收只需传回结果目录：

```bash
tar czf restad_results.tar.gz -C RESTAD/thesis/results $(ls -t RESTAD/thesis/results | head -1)
```

再 `scp` 拉回本地即可。结果目录里的 `pairwise_*.md` 就是能直接往论文里抄的表，
`figures/*.png` 是图。

> 结果目录里的 `model.pth` 和 `scores.npz` 是复核证据，`tar.gz` 会比较大（约 1 GB）；
> 只要表和图的话，把这两类文件排除掉，几十 KB 就够了。

## 六、常见问题

**`bash: $'\r': command not found`**
脚本被 Windows 换行污染了。执行：`sed -i 's/\r$//' RESTAD/thesis/deploy/*.sh`

**`CUDA out of memory`**
并发开太多。把 `WORKERS` 降到 4 或 3，重新执行 `run_matrix.sh`。
并发开太多。把 `WORKERS` 降到 4 或 3，重新执行 `run_matrix.sh`。
注意重跑会把同名的 run 目录覆盖掉，所以中途停之前先记下已完成的目录名；
如果只想补齐缺的，把 `SEEDS` / `VARIANTS` / `DATASETS` 缩小到缺的那部分。

**`No module named hydra`**
`setup_env.sh` 没跑成功，或换了别的解释器。用 `PYTHON=/root/miniconda3/bin/python bash ...` 指定。

**中途断了怎么办**
已经完成的 run 目录是完整的，不会损坏。直接重跑 `run_matrix.sh` 即可，
它会新建一个结果目录并重跑全部任务；如果只想补齐缺的，把 `SEEDS`/`VARIANTS` 缩小到缺的那部分。

**结果和本地对不上怎么办**
不同显卡/CUDA 版本的浮点运算顺序不同，指标会有千分之几到百分之一量级的差异，属正常。
**论文只用一台机器的结果**，不要混用本地和服务器两次跑出来的数字。
