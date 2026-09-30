# Quest 开发、SDSC Expanse 执行

本工作流把 Quest 保留为唯一源码编辑位置，通过已有 SSH 共享连接向
SDSC 部署独立快照，并按需查询 Slurm。Codex 会话始终留在 Quest。
SDSC 不安装 Codex、VS Code Server 或常驻工作流服务，也不申请用于
维持编辑器在线的交互作业。

最初的工具、连接检查和干运行阶段已经完成，当时未提交计算作业。
用户随后明确授权本任务上传源码并执行**一次**首次 GPU smoke，
资源为 1 H100、4 CPU、16 GiB、最长 5 分钟。该授权已用于作业
`54345483`，测试成功且结果已回传；不要再次执行提交步骤。
该次测试范围为同步、GPU 可见性、微型计算、结果持久化和回传，不能
替代科学协议和执行安全认证。已有提交应继续检查与回传，不能因为
切换会话再提交一次。

用户随后已明确授权**启动正式训练，并按工作负载选择多 GPU 资源**。
该授权包括必要的环境、模型和数据准备、存储及分布式预检，以及通过
科学门槛后的 Qwen3-v2 训练；无需重复请求授权。当前默认顺序为 G0
通过后启动 seed-42 pilot，不自动扩展为三种子全因子或 Gemma。
授权不允许绕过失败门槛、伪造旧平台认证或重复未知状态的提交。

正式训练准备的最新状态以 `docs/refactor/current_handoff.md` 为准。
下文首次 smoke 的资源限制和单次授权是历史记录，不是当前训练上限。

## 当前 teacher 适配与验收

2026-09-28 的当前路径是独立训练并验收一个带真实权重哈希的新 teacher。
单卡适配预检 `54493015` 和修复后的四卡执行预检 `54494477` 已完成；
完整适配作业 **54494742** 已 COMPLETED / 0:0，实际用时 1 小时 19 分 36 秒，
使用 4 H100、24 请求 CPU、192 GiB 工作负载预算、最长四小时。
四个开发检查点都通过八项门槛，按固定规则选择第 128 步；其答案正确率
100%、严格证明正确率 511/512。产物已持久化并回传小型报告，不重复提交。
资格作业 **54496291** 也已 COMPLETED / 0:0，实际用时 1 小时 15 分 14 秒。
两个 128 例评估均通过八项门槛，2,048/2,048 个候选验证通过并覆盖全部
256 个提示。独立 CPU 复核作业 **54497294** 已 COMPLETED / 0:0，用时
41 分 18 秒，全部原始证据重放与永久阶段记录检查通过。独立审查者随后
签发正式验收，最终产物为 **`formal_teacher_accepted=true`**。四个验收
文件已在 SDSC 独立持久目录保存，并经只读回查确认哈希一致。不要重复
提交、签发或发布这些已完成任务。

验收适用于已独立审核的 **teacher 专用 256-token 协议**，八项数值阈值
未降低；不声称原 128-token 协议通过。新的 student 入口为
`qwen3-v2-adapted-preflight` → `qwen3-v2-adapted-calibration`，使用已验收
权重的真实哈希和原始九文件示例证据，保留全部 student 训练参数。
历史 v1–v5 两阶段均为 2 H100 / 24 CPU / 192 GiB。2026-09-30 用户授权
内存修复后的重提；已独立接受的 v6 使用 2 H100 / 24 CPU / 384 GiB，两阶段分别
最多 1 小时和 2 小时，仍要求32 GiB与20%两项内存余量门槛。新版本的匹配预检 **54547547** 已提交并运行，Quest quser43的有限监控
会在严格通过后自动提交一次校准；不追认旧失败任务，也不改 teacher 历史环境。
必须先完成独立协议接受，再以实际新预检的成功报告允许校准；不能重启旧
v3 流程，也不能把校准当作完整 G0。实际提交状态读取本地回执及 current
handoff；操作说明见 [适配 teacher 的 student 校准](refactor/sdsc_adapted_student_calibration_20260928.md)。

用户另已授权监控修复后的预检 `54506703`，验收成功后自动提交一次
student 校准训练。独立快照重放、有限 Quest 监控、防重复提交及连接中断
恢复规则见 [student 自动接续](sdsc_student_automation.md)。先检查实际
supervision 状态；不要重启历史 teacher/G0 流程，也不要把校准当作完整
正式实验的全部阶段。

固定训练为 8,192 条独立训练数据、512 次优化更新，在四个预定检查点
上完整评估独立的 512 条开发数据。后续 `qwen3-v2-teacher-qualify` 只接受
开发集上第一个满足八项门槛的检查点，要求先完成真实 Git 实现提交及
独立 review-only 接受提交。它再依次测量 32×8 训练探针、原验证集
第 128–255 行、原前 128 行和全新的 256×8 teacher store。
任一门槛失败即停止；不重采样、不换检查点重新试补充验证集。
完整规则见 [teacher 适配协议与验收](refactor/sdsc_teacher_adaptation_20260928.md)。

执行成功与独立科学验收分开记录。默认 `fetch` 仍只取有限报告和日志，
不取回权重。独立 CPU 审计通过相同 SSH master 提交一个有界批处理作业，
在计算节点的本地源码副本上检查真实原始文件、永久阶段记录与全部指标。
审计程序本身不提交 Slurm、不运行推理，也不自行签发 teacher 接受结论。
不要在登录节点执行完整审计：当前冻结实现有数十分钟的重复 CPU 校验。
实查 `nwu181` 没有 CPU 分区权限，shared QoS 最少要求一张 GPU；已通过
预检的最低资源为 **1 H100 / 1 CPU / 16 GiB / 90 分钟**，计算时隐藏 GPU。
本次审计已在四卡资格作业结束后唯一提交并完成，真实回执与验收证据
均已保留。以下命令仅供后续按需只读复查，不需重新运行审计或下载权重：

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
/usr/bin/python3.12 -I -B .sdsc/diagnostics/teacher-adaptation-v1/batch_audit_observe.py status \
  --intent-dir /gpfs/projects/p32737/del6500_home/OPD/.sdsc/fetched/54496291/batch-audit-b04f8e93a2cc7f3e2961de65ecdac7ca
/usr/bin/python3.12 -I -B .sdsc/diagnostics/teacher-adaptation-v1/publish_independent_teacher_acceptance.py check \
  --accepted-dir /gpfs/projects/p32737/del6500_home/OPD/.sdsc/diagnostics/teacher-adaptation-v1/accepted-teacher-54496291-4641d1f2c54e627e \
  --inventory-sha256 8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed
```

两者仍先检查当前 Quest 主机的共享 SSH master；认证失效即停止，不尝试
密码登录。审计 observer 保留原始 accounting，并用实际已保存的 live
`scontrol` 绑定处理 SDSC 不保留 `sacct Comment` 的情况；不要再用旧审计
提交工具的 strict-Comment 查询入口。

## 历史教师失败修复与诊断

教师准备作业 `54345715` 已失败：2048 个候选仅 8 个通过，覆盖 4/256 题。
主要观察是复制提示中的伪示例；原验证器的完整覆盖门槛不能降低。
修复与证据见 [教师提示修复记录](refactor/sdsc_teacher_prompt_repair_20260918.md)。

新的 `qwen3-v2-teacher-prompt-probe` 是有界对照诊断，资源固定为
1 H100、24 CPU、192 GiB、最长 30 分钟。已提交的真实作业是
**54351516**；不要重复提交。初始状态是维护预留导致的 PENDING，
调度估计为 9 月 20 日；更短时限和更小资源并未提前预计日期。

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
tools/sdsc status 54351516
tools/sdsc logs 54351516 --lines 60
# 作业产物持久化之后：
tools/sdsc fetch 54351516
```

只读监控已在 Quest quser34 启动，PID `2037247`，记录位于
`.sdsc/supervision/probe54351516-readonly-v1/`。每五分钟查询一次，最多
十四天；终态后自动取回小型结果。它不会提交、取消或重试作业，也不会
自动接受科学结论或启动训练。共享 SSH 失效或固定控制文件变化即停止。
新会话先读现有 `state.json`，不要重新创建同一作业的监控。

新诊断确有必要时才创建一个全新的只读监控计划：

```bash
tools/sdsc_probe_watch prepare --job-id NEW_EXISTING_JOB_ID --flow NEW_FLOW
tools/sdsc_probe_watch run --plan /absolute/Quest/path/to/NEW_FLOW/plan.json
```

该入口只接受已提交且有匹配本地回执的 prompt probe；`prepare` 不是
Slurm 提交。停止后的 flow 不会自动重新启动，先核对记录和 SSH 状态。

探针对固定前 32 题分别生成旧提示 candidate 0、新提示 candidate 0–7。
公平的单次采样比较使用 `paired_candidate_zero`；八次采样覆盖率单独报告。
诊断 `passed=true` 仅表示执行和结果保存成功，不能代替正式 teacher
store、完整覆盖或 G0。新 prompt-v4 协议目前仍为 proposed。
旧 teacher→calibration 和 G0→pilot 流程都已停止，不能直接重新启动：
新科学版本需要完整教师数据、匹配的预检以及一致的新 HEAD／协议绑定。

## 当前配置与边界

| 项目 | 值及来源 |
| --- | --- |
| Quest 源码 | `/gpfs/projects/p32737/del6500_home/OPD`，当前工作区真实路径 |
| SDSC SSH | `zgao12@login.expanse.sdsc.edu`，用户提供 |
| 远端项目根 | `/home/zgao12/quest-runs/OPD/`，仅用于源码和小型控制信息 |
| Slurm account | `nwu181`，已通过远端 account 关联及额度查询核实 |
| Partition | `nairr-gpu-shared`，已查询远端分区配置 |
| QoS | `nairr-gpu-shared-normal`，账户与分区共同允许的值；现场核查修正了最初提供的 `nairr-gpu-shared` |
| GPU 类型 | `h100` |
| 额度查询 resource | `expanse_nairr_gpu`；不能用作 `sbatch --account` |
| 共享连接 | 运行时计算 `$HOME/.ssh/cm/sdsc-$(hostname -s)` |
| 首次测试资源 | 1 张 H100、4 CPU、16 GiB 主机内存、最长 5 分钟 |

工具从自身位置确定项目根，不依赖启动目录或终端里未导出的变量。
Quest 已确认 `/usr/bin/python3.12` 为 Python 3.12.13；默认 `python3`
则为 3.6.8。入口使用已发现的 `python3.12`，只需要标准库，不激活
项目 venv；需要时可显式设置 `SDSC_LOCAL_PYTHON`。远端控制 Python
通过连接现场发现，最低 3.8；微型 GPU 运行时最低 3.9，并需已有
支持目标 GPU 的 PyTorch。这些最低要求不改变正式项目的依赖锁。
原有 `AGENTS.md`、ServerScheduler 集成、Slurm 脚本和环境配置保留。
这里是用户明确要求的独立 Expanse 执行路径；不会恢复或调用历史
Quest Slurm 提交脚本，不修改外部 ServerScheduler。

## 正式训练的分阶段入口

正式输入和保留结果使用已发现的项目目录：
`/expanse/lustre/projects/nwu181/zgao12/OPD/`。作业 `54345521` 已在真实
H100 节点 `exp-19-07` 上完成 Lustre 写入、读取和回传，Slurm 为
`COMPLETED / 0:0`；没有添加 `--constraint=lustre`。每个后续作业仍在
自己的节点上验证挂载和读写。项目组 quota 为 50 TiB，属于整个组，
不是本人的独占容量。HOME 不保存训练数据或 checkpoint。

固定模型已直接下载到项目下 `cache/huggingface`，21 个文件共
20,476,854,927 bytes；权重校验发布方 LFS SHA256，其他文件校验 Git
blob SHA1，并记录本地 SHA256。revision 仍为仓库冻结的 Qwen3-1.7B
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` 与 Qwen3-8B
`b968826d9c46dd6066d109eabc6255188de91218`。作业只复制这两个快照到
节点本地盘，离线读取，不重复下载。

正式 runtime 在 `envs/qwen3-v2-g0-py31213-cu128-v1`。准备过程复用
SDSC 的 Conda 管理器，创建 Python 3.12.13 并安装仓库 G0 依赖锁，
包括 `torch==2.8.0+cu128`。不能只根据目录存在判断安装完成：先读取
项目 `bootstrap/runtime-prepare-status.json` 的 `complete` 状态和
`runtime-identity.json`，再通过 `tools/sdsc check --python ...` 核实。
完整安装日志与 freeze 留在该 `bootstrap` 目录。

| 任务 | 明确资源 | 成功所代表的范围 |
| --- | --- | --- |
| `qwen3-v2-preflight` | 2 H100、24 CPU、192 GiB；本次计划 30 分钟 | 真实模型、NCCL、global-64 单次更新、完整状态保存恢复、内存和持久化 |
| `qwen3-v2-teacher-prepare` | 1 H100、24 CPU、192 GiB；最长 4 小时 | 完整冻结数据集与 256 prompts × 8 候选，formal teacher store 质量及哈希验证 |
| `qwen3-v2-g0-calibration` | 2 H100、24 CPU、192 GiB；当前显式最长 1 小时 | 原始 canonical-SFT 训练及真实 checkpoint/metrics 校验，不代表完整 G0 |

共享分区现场限制每作业最多 3 张 GPU；四卡分区已现场确认是
`nairr-gpu`，账户允许的 QoS 为 `nairr-gpu-normal`。当前教师生成实现是串行，增加 GPU 不会加速这个阶段。
30/60/120 分钟预检的远端 `sbatch --test-only` 比较记录在
`.sdsc/preflight-scheduler-estimates.txt`；这些估计中的编号不是已提交
计算作业的回执。实际提交只接受 `sbatch --parsable` 的真实回执。

这些任务都明确输出 `g0_passed: false` 和
`execution_class_certified: false`。H100 不符合旧 Blackwell 证书的
硬件范围，不能伪造 ServerScheduler context 来复用其通过结论。
完整 G0 仍要做 readiness、probe/circuit、anti-shortcut、真实恢复和
最终科学判定；只有有效 G0 通过后才允许后续 seed-42 pilot。

正式数据准备还需要独立 Git provenance，因为原科学代码检查真实
干净 HEAD 和 accepted review 祖先。`tools/sdsc_provenance.py create`
从已确认的 public HEAD 导出受限 `history.bundle`、manifest 和 wrapper
清单，拒绝科学目录的未提交变更/隐藏 source shadow。它不复制原
`.git` 配置、hooks、reflog 或凭证，不强制提交本地修改。dirty tools/docs
仍完整记录在 wrapper 快照；恢复的科学 checkout 与 wrapper 分开。

先对新源码 `sync --dry-run`、`sync`，再用该 run record 创建并传输
provenance（以下变量均须来自本次真实回执）：

```bash
/usr/bin/python3.12 -I -B tools/sdsc_provenance.py create \
  --wrapper-manifest ".sdsc/runs/$RUN_ID.json"
/usr/bin/python3.12 -I -B tools/sdsc_provenance_upload.py \
  --artifact "$ARTIFACT" --manifest-sha256 "$PROVENANCE_SHA" \
  --run-id "$RUN_ID" --dry-run
/usr/bin/python3.12 -I -B tools/sdsc_provenance_upload.py \
  --artifact "$ARTIFACT" --manifest-sha256 "$PROVENANCE_SHA" \
  --run-id "$RUN_ID" --upload
```

外部记录的 `PROVENANCE_SHA` 与源码 hash 必须进入提交 intent、回执和
作业参数，不能从远端不可信目录自己算一个 hash 再当作信任依据。
teacher 提交另需 `--hf-home`、`--provenance-dir` 与
`--provenance-manifest-sha256`，目录严格为
`/home/zgao12/quest-runs/OPD/provenance/<PROVENANCE_SHA>/`。
先运行 worker 的 `--validate-only` 校验科学绑定再分配 GPU；该模式不
加载 tokenizer/model 或生成数据。`output_root` 必须含 `qwen3-v2`
路径组件，这是原科学配置的要求。

teacher sampling seed 保持 **31415**，实验 seed 是 **42**；完整七个
数据 split 共 **144000** 条，不因 `task.num_examples=256` 缩小。
原生成器只有全部 2048 候选结束后才写 ledger，中途终止可能没有
部分 ledger，不能声称可以断点恢复。失败时仍保留已有数据、诊断
和日志。fetch 只回传小报告；数据、权重、checkpoint 保留在项目存储。

### 当前真实作业与训练接续

双卡预检 **54345604** 已在 exp-19-04 成功完成，用时 2 分 48 秒，
Slurm 主作业和子步骤均 `COMPLETED / 0:0`。它实际验证了两个固定模型、
NCCL、FSDP FULL_SHARD、global-64 更新及模型/optimizer/scheduler/RNG
完整恢复；有限 cgroup 上限 192 GiB、峰值约 70.47 GiB，GPU 最大 reserved
分别约 44.89/29.79 GiB。10.95 GB checkpoint 已持久化，未下载到 Quest。

教师作业 **54345715** 已正式提交并开始运行；完整数据集已生成，模型
已加载，数据质量仍待全部候选完成后确认。不得因为切换会话重复提交。

2026-09-18 02:36 UTC，有限接续进程已在 Quest `quser44` 启动，PID
`1192904`，状态为 `waiting_teacher`。记录目录为
`.sdsc/supervision/teacher-54345715-to-calibration/`，新会话先读其中
`plan.json` 和 `state.json`；PID 与作业状态都是需重新核查的现场信息。
校准源码 release 为 `20260918T023430Z-91bb7848d574-73d8d61d`；正式
参数已固定为 2 H100 / 24 CPU / 192 GiB / 1 小时。30/60/90 分钟的
远端 test-only 都估计可立即开始，选择 1 小时为首次完整校准的评估、
checkpoint 与持久化保留余量。此时校准尚未提交，没有真实 job ID。

```bash
tools/sdsc status 54345715
tools/sdsc logs 54345715 --lines 60
```

校准训练需要新 release/provenance 和显式 `--teacher-job-id 54345715`
及 `--preflight-job-id 54345604`。`submit --dry-run` 只生成参数；其中
prerequisites SHA 是占位符，不表示已通过上游验证。正式提交会在远端
读取两项作业的真实记账、原始报告、发布回执和源码清单，核对 teacher
数据全部实际字节；只有成功且没有其他活跃 OPD GPU 作业才提交。
工作节点再次验证这份 proof，并复制输入到本地盘。失败报告不是可恢复
checkpoint；不能从半写入文件恢复或自动重试。

`tools/sdsc_supervise --plan .sdsc/supervision/FLOW/plan.json --authorize`
可在 Quest 启动**单次、八小时后停止发起操作**的接续进程，五分钟查询一次：
确认预检通过 → 等教师完成并验证 → 回传小报告 → 提交一次双卡校准 →
检查终态和产物 → 回传小结果后退出。计划固定真实 Quest 路径/主机、
release 内容 SHA、控制工具 SHA、资源、模型/结果位置及两个上游 job ID。
它不在 SDSC 部署服务，也不分配 Quest 或 SDSC 交互算力。
已开始的控制命令最多等待 420 秒；达到期限不会取消远端作业。

状态保存在相邻 `state.json`，包含 PID、最近状态、真实校准 job ID 和
提交回执。新会话先读这个状态及 `.sdsc/submissions/`，不能再次启动同一
流程。活动期间修改已固定的控制工具会让接续停止，避免运行未审阅代码。
SSH 失效立即停止，不认证重试；失去 sbatch 回执只进入 reconcile，不能
重提。用户手动恢复 SSH 后也必须先核查原状态，不能盲目重新启动进程。
若教师质量不合格，接续保存小型诊断并停止，不启动训练。校准成功仍不
代表 G0 通过，不会自动启动 pilot、三种子全因子或 Gemma。

完整后继现在由独立工具 `tools/sdsc_pipeline` 执行，保留上述旧进程。
实际启动状态以 `.sdsc/supervision/teacher54345715-g0-pilot-v1/` 的
`launch.json`、`state.json` 和真实进程为准；源码存在不是已经启动。
实现保留原科学 checkout，分别验证冻结配置与真实初始 checkpoint SHA，
通过私有路径映射运行两次真实续训，再进行完整 G0 终验和重放。
详见[自动接续与验收边界](sdsc_automation_gates.md)。

正式路径新增后，244 项 CPU 测试、Ruff、Shell 语法和 diff 检查通过。
独立审查还验证了真实 clean checkout 的配置绑定、原 train dry-run、
实际双卡报告，以及控制面 proof 到作业输入暂存的接口。测试不代替真实
训练结论。校准作业使用 `--signal=B:TERM@300`，留出停止训练和持久化
的时间；只有稳定文件及读回哈希通过才宣告发布成功。

## 完整自动接续

本流程已于 2026-09-18 03:54 UTC 在 Quest `quser44` 启动，PID `1693084`。
部署 run 为 `20260918T035115Z-7d6f46dcb09a-1e6ddd7b`，启动后实测状态为
`waiting_calibration`，上游仍为 `waiting_teacher`；这不代表 G0 或 pilot
已经通过。源码快照、来源证明和两端计划已验证，380 项 CPU 测试通过。
进程和作业状态会变化，新会话必须使用下面的状态命令重新检查。

后继进程始终在 Quest。它等待既有 teacher→calibration 进程报告成功，
重新核对真实 Slurm 终态与持久化证明，然后依次执行以下阶段。所有阶段
固定 24 CPU、192 GiB；每次提交前用相同脚本和资源比较所选时长与一个
较长候选的 `sbatch --test-only`，保存估计，再单次 `sbatch --parsable`。

| 阶段 | GPU 与最长时间 | 执行内容 |
| --- | --- | --- |
| G0 | 2 H100，12 小时 | 复用真实校准、两次 step20 恢复、readiness/probes/circuits/anti-shortcut、完整终验与重放 |
| 四卡预检 | 4 H100，30 分钟 | 原模型、NCCL、global64、完整状态扰动/恢复及内存余量 |
| prepare | 1 H100，30 分钟 | 固定 seed42 的完整 pilot manifest 与上游验收 |
| teacher-shard | 16 个任务，每项 1 H100、4 小时，最多并发 4 项 | 4096 prompts × 8 候选，原每候选 RNG，不降低 coverage |
| pilot-inputs | 1 H100，24 小时 | 原序合并全部 ledger、4096-prompt common bank 与 teacher scoring |
| train-cell | 8 个任务，每项 4 H100、12 小时，逐项运行 | 六种 factorial 方法及 SFT、GRPO anchors，原四卡 batch 语义 |
| initial-circuits / final-circuits | 1 H100，6 / 8 小时 | 真实训练数组验收，4 个初始与 32 个终态 circuit/exact-patching |
| local-fork / resume / dynamics / finalize | 1 / 4 / 1 / 1 H100，12 / 2 / 2 / 2 小时 | 原 local fork、两次真实恢复、32 个 dynamics 和完整 pilot 终验 |

这些时长是无同类完整历史时的保守上限，不是实际运行时间或成功保证。
下一个阶段只在前一个阶段真实 `COMPLETED / 0:0`、所有 step 成功及
持久化科学报告通过后提交。发现其他 pending/running OPD GPU 作业时
停止新提交，两个数组最多同时占用四张 GPU。没有提交完整三种子 factorial
或 Gemma 的路径。

G0 接受器保留 26 项原科学检查，明确以 SDSC 单次执行证据和已审阅适配器
身份替换两项旧 backend/certificate 检查。旧科学文件和证书字节不改；
输出 `execution_class_certified=false`。pilot 的新 4096-prompt 输入单独
绑定原 G0 报告和实际字节，不假装它与旧 256-prompt bank 是同一个文件。
最终验收沿用原科学检查，并显式记录这个输入绑定适配。

后继代码及入口：`sdsc_pipeline.py`、`sdsc_pipeline_remote.py`、
`sdsc_pipeline_job.py`、`sdsc_pipeline_storage.py`、`sdsc_g0.py`、
`sdsc_resume.py`、`sdsc_finalize_g0.py`、`sdsc_pilot_preflight.py`、
`sdsc_pilot.py`（均位于 `tools/`）。`sdsc_pipeline_prepare.py` 只创建及部署
已审阅计划，不提交；启动入口会核对本机、仓库、所有固定控制文件 SHA。
新会话先检查，不能重新启动已经有 `state.json` 的流程：

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
tools/sdsc_pipeline --plan .sdsc/supervision/teacher54345715-g0-pilot-v1/plan.json --status
```

远端小型 plan/claim/receipt 在 `/home/zgao12/quest-runs/OPD/pipelines/FLOW/`；
源码仍是独立 `releases/RUN_ID/`；持久结果在
`/expanse/lustre/projects/nwu181/zgao12/OPD/pipeline-results/FLOW/STAGE/INDEX/`。
节点复制及逐文件 SHA/read-back 完成后才运行；每个阶段只发布新增或变更
文件，旧 checkpoint 始终保留。每个新 job 的输入清单绑定实际存储路径和
文件 SHA，节点剩余空间必须覆盖所选输入加 256 GiB 余量。不能因此推断
未来所有训练输出必定放得下；容量不足时停止，保留远端已完成结果。

实际 `sacct` 文本同时记录数组 `JobID` 与数值 `JobIDRaw`，不伪造数组记录。
二者区别见 [Slurm sacct 官方说明](https://slurm.schedmd.com/sacct.html)。
完整 checkpoint 留在 Expanse；每阶段完成或失败后自动回传有大小上限的
报告和日志尾部到 `.sdsc/fetched/pipeline-FLOW/`，不覆盖 Quest 源码。

新运行时位于 `envs/qwen3-v2-pilot-trl0222-overlay-v1`：同一 SDSC 主机上
用已验证 G0 Python 建立独立 `--copies --system-site-packages` venv，
额外固定 `trl==0.22.2`，仍逐项验证原 19 个依赖。容器只提供私有 mount
namespace，真正运行此 host Python；其 G0 base 也显式只读 bind。环境
准备证据以 `bootstrap/pipeline-v2/environment.json` 的完整验证为准。
登录节点准备显式限制 OMP/MKL/OpenBLAS/NumExpr 为一线程；曾出现的默认
多线程导入长期未结束，限线程对照已成功。计算作业也显式设置每 rank
的数值库线程上限，避免继承登录节点默认值。安装后验证若中断，先检查
原准备进程已退出和已有目标，再使用 `--verify-prepared` 只做完整复验；
它不会重复安装、下载或把部分安装当作通过。
MIB 固定在 `b759df34433c9e31043ba9e02908ce0bf20e894f`，含真实 submodule
revision 和逐文件 SHA。模型沿用原离线 cache，不重新传模型或大数据。

进程最多运行十四天、每五分钟查询一次，退出 Codex 会话不影响已独立
启动的 Quest 进程或 Slurm 作业。前提是 Quest 主机和 SSH master 仍存活。
当前普通终端的前台 master 若随终端退出，按用户规则立即停止，不能自动
认证。未来手动认证可用 `ssh -fMN` 让认证成功后的 master 转入后台；不要
为此结束当前有效连接，也不要读取凭证。科学失败、未知回执、SHA 改变、
三次记账不明或 SSH 失效都会停止推进，不会自动重复提交或降低门槛。

提交回执不明时，只核查现有 claim：

```bash
tools/sdsc_pipeline --plan .sdsc/supervision/teacher54345715-g0-pilot-v1/plan.json --reconcile-stage STAGE
```

恢复认证不会自动重启已经停止的控制进程。先核查保存的阶段、所有真实
job ID 和持久化结果，再从未完成的边界恢复；不得删除 claim 来重新提交。

## 认证与连接检查

用户在**同一台 Quest 主机**的普通终端手动认证。不同 Quest 主机的
`hostname -s` 不同，因此其控制连接不能互相代用。工具先执行
`ssh -O check`；连接缺失或失效时立即停止，不循环重试、不启动密码
或验证码交互。

普通终端建立连接的示例：

```bash
mkdir -p "$HOME/.ssh/cm"
chmod 700 "$HOME/.ssh/cm"
ssh -F /dev/null -M \
  -o ControlPath="$HOME/.ssh/cm/sdsc-$(hostname -s)" \
  -N zgao12@login.expanse.sdsc.edu
```

保留该普通终端的 SSH 进程。上述 `-F /dev/null` 避开本次 Quest
系统 SSH 配置的权限错误；不关闭主机密钥检查，仍使用正常的
`known_hosts` 校验。首次主机密钥确认和认证由用户在普通终端完成。
不向 Codex 提供密码、验证码、私钥或 `auth.json`。

工具的远端操作使用相同 `ControlPath` 和 `BatchMode=yes`，只复用
用户已经认证的连接。命令和参数进行安全传递，不使用 `eval`。
`sbatch`、`squeue`、`sacct` 和 `scancel` 均通过 SSH 在 SDSC 运行，
从不在 Quest 直接执行。

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
tools/sdsc check
tools/sdsc sync --dry-run
```

`check` 检查实际远端身份、命令和路径，发现已有的软件环境，并记录
登录节点可见的存储信息。它不申请计算节点，也不把登录节点检查结果
当作 H100 节点的 CUDA 或文件系统验证结果。必要命令不可用、身份不符
或路径检查失败时，先解决该具体问题，不绕过检查提交。
若命令存在，还会执行有时限的只读 account/QoS 关联与额度查询；
查询失败会保留未验证状态，不安装工具、不交互输入凭证。本次已经
发现并核实了下列路径，可重复检查包版本与登录节点路径权限：

```bash
tools/sdsc check --python /usr/bin/python3.11 \
  --result-root /home/zgao12/quest-runs/OPD/smoke-results \
  --container-runtime /cm/local/apps/singularitypro/4.1/bin/singularity \
  --container-image /expanse/projects/qstore/installs/containers/singularity/Expanse-Air/pytorch/pytorch-nvcr-25.03.sif \
  --container-python /usr/bin/python
```

包版本检查读取安装元数据，不导入 PyTorch、不初始化 CUDA。容器检查
不使用 `--nv`。这仍不能证明 H100 节点可以访问这些路径；尚未创建的
结果目录会如实报告 `exists: false`，不会因此自动创建目录或启动作业。

## 源码快照

先运行 `sync --dry-run` 审阅完整文件清单、逐文件大小和总大小，
计划保存在 `.sdsc/last-preview.json`。之后才运行 `sync`；工具要求
当前内容哈希与已审阅计划一致，发生修改时需重新 dry-run。
`sync --dry-run` 生成新的 `run_id`，成功的 `sync` 沿用该计划 ID 并在
上传回执中返回；以成功回执确认部署完成。远端源码放在
`releases/<run_id>/`；后续本地编辑需要新快照。排队或运行中的作业
引用原来的目录，不能覆盖该目录来“更新”作业。
预览同时记录源码总字节数和含清单的 tar 总字节数。文件来源是 Git
已跟踪文件及未被忽略的新文件，再应用源码目录／类型白名单；额外
排除原因记录在计划的 `excluded` 字段。单文件上限 2 MiB，源码
总量上限 48 MiB，所有符号链接均跳过。必要文件若被排除，先检查
清单并调整明确的源码规则，不通过复制环境或数据目录绕过限制。

快照以当前工作区真实文件内容为准，包含符合同步规则的未提交修改和
必要新文件。逐文件 SHA-256 与快照内容身份记录在清单中；Git HEAD
仅是辅助来源信息，不能代替真实文件哈希。部署快照身份不写入科学
输入，不改变现有科学配置、产物或执行安全指纹的定义。

只传代码、配置、文档、测试及必要小文件。排除版本库元数据、
`.codex`、`.ssh`、凭证、环境文件夹、模型权重、大数据、缓存、
checkpoint、历史输出和本工具的本地状态／回传结果。不跟随指向
项目外部的符号链接，不使用 `rsync --delete`。被排除的文件应先在
dry-run 结果中检查；不要通过改名绕过凭证或体积限制。

工具不强制 commit、reset、checkout 或清理工作区。独立快照允许
开发中的修改获得真实内容身份，不要求为了传输产生 Git 提交。
大型模型和数据另行制定传输计划，不经过这个源码通道反复传输。

## 环境发现与复用

先通过 `check` 发现 SDSC 已有 Conda／Python／容器工具与可用环境。
已有合适环境时优先复用；提交参数必须明确指定已经核实的运行时。
不复制 Quest 的 Conda 或 venv，不根据本地环境名猜测远端环境名，
也不自动加载猜测的 CUDA module。

仓库 `environment.yml` 描述 `posttrain-circuits` 与 Python 3.12.11；
`pyproject.toml` 固定了 PyTorch 2.8.0、Transformers 4.56.2 等科学
依赖。这些是仓库声明，**不是 SDSC 已安装环境的证据**。首次微型
GPU 测试只需要测试脚本明确使用的依赖；成功后仍需为正式 OPD 实验
单独验证完整固定运行时、模型 revision、离线缓存和数据契约。

本次只读发现的运行时如下：

| 运行时 | 现场证据与用途 |
| --- | --- |
| 宿主 `/usr/bin/python3.11` | Python 3.11.5，无 PyTorch；用于控制、源码暂存与结果持久化 |
| `anaconda3/2021.05` | 临时加载 module 后 Conda 4.10.1、Python 3.8.8、NumPy 1.20.1；只发现共享 base，无 PyTorch，不选作 GPU 运行时 |
| `/cm/local/apps/singularitypro/4.1/bin/singularity` | SingularityPRO 4.1.2，已成功读取现有镜像内的包元数据 |
| `pytorch-nvcr-25.03.sif` | SDSC 当前 H100 示例指定的共享镜像；内置 `/usr/bin/python` 3.12.3、PyTorch `2.7.0a0+7c8ec84dab.nv25.3`、NumPy 1.26.4 |

镜像绝对路径见上面的 `check` 命令；它来自实际读取的
`/cm/shared/examples/sdsc/ExpanseAIR/pytorch/run-gpu-nvcr.submit`，不是
猜测或新下载的镜像。该示例目录也列于
[SDSC 官方 AI 资源说明](https://www.sdsc.edu/systems/expanse/user_guide.html)。
此容器仅用于基础设施 smoke，不符合正式 OPD 的 PyTorch 2.8.0 固定
依赖，不能据此直接开展正式实验。

容器选项必须成组提供：`--container-runtime`、`--container-image`、
`--container-python`；`--python` 仍指宿主 Python。作业在宿主验证本地
工作盘、暂存源码与保存结果，仅把微型 GPU 计算放入容器，通过 `--nv`
使用分配的 GPU，原样保留 `CUDA_VISIBLE_DEVICES`。镜像不复制到 Quest
或节点本地，不自动拉取。若不选容器，则显式宿主 Python 必须已有
可用的 GPU PyTorch。

检查记录绑定镜像路径、大小 `11381182464` 字节和修改时间
`1754055734887617455` 纳秒；提交和执行前后校验这些属性，变化时拒绝。
这是共享镜像的变更检查，**不是镜像内容哈希**。未读取全部 11.38 GB
镜像来计算哈希，正式实验仍需独立固定完整运行时。源码则始终使用真实
逐文件 SHA-256。容器身份进入提交 intent、回执与结果验证。

本工具不安装环境或下载容器／模型。若发现的运行时缺少依赖，先报告
具体差异并准备独立环境方案。不能通过修改科学依赖来让 smoke 测试
“看起来通过”。登录节点不运行 GPU 运算。

## 存储与作业生命周期

HOME 只放部署源码和小型控制信息，不能作为训练工作盘。作业启动后
验证实际节点本地目录，再把源码快照复制进去并验证内容哈希，所有
微型计算在该节点本地副本上执行。节点临时盘不是持久存储。

SDSC 官方指南的 AI resource 部分说明 H100 使用
`--gpus=h100:1`，节点本地目录为 `/scratch/$USER/job_$SLURM_JOBID`，
作业结束会清理；该部分说明 AI 节点不挂载 Lustre，长期数据使用
Ceph。指南的通用 Lustre 部分另有 `--constraint=lustre` 要求。
因此本工具不自动添加该 constraint，也不从登录节点可见性推断
H100 可访问性。[SDSC Expanse 用户指南](https://www.sdsc.edu/systems/expanse/user_guide.html)

正式实验前必须取得**目标计算节点上**的证据，确认：

- 输入的真实位置、访问权限、版本／哈希以及节点可达性；
- 已有运行时和必要共享库在计算节点可用；
- 实际本地工作目录可写、容量满足实验需要；
- 持久输出位置可写、不是临时盘，并具有明确的保留策略；
- 若采用 Ceph/S3，端点、目标和传输工具已准备好，认证由用户管理。

当前 `--result-root` 接受现有 POSIX 目录，不接受 `s3://` 地址，
也没有实现 S3 上传器。正式实验若使用 Ceph 对象存储，需要另外实现
并验证输入传入和结果持久化适配；不能用一个登录节点目录代替验证。

第一次 smoke 可以显式选择
`/home/zgao12/quest-runs/OPD/smoke-results` 保存少量 JSON／日志。
这是本工具唯一允许的 HOME 结果位置，仅用于这次基础设施测试的小型
证据；`sync` 会创建这一空目录，之后应重新 `check` 检查权限，提交
仍须 `--storage-confirmed`，
且作业在实际 H100 节点上检验挂载和读写。不得据此把 HOME 选为
正式训练数据或 checkpoint 存储。若持久位置在计算节点
不可见或写入失败，测试必须失败，不能只留下 TMPDIR 中的“成功”。
需保留的结果在结束前写入已确认的持久位置，再从 Quest 按需取回。
TIMEOUT、节点失败或强制终止可能阻止收尾，所以不能仅靠退出 trap
保证正式训练结果；正式工作流还需要周期性、经验证的持久化策略。

## 命令入口与提交边界

`tools/sdsc --help` 和各子命令 `--help` 是参数的直接说明。

| 命令 | 行为 |
| --- | --- |
| `check` | 检查已有连接、身份、必要命令、路径和软件环境 |
| `sync --dry-run` | 本地生成同步计划，展示文件和大小，不上传 |
| `sync` | 上传已预览的独立快照，返回同一 `run_id` 的成功回执 |
| `submit RUN_ID --dry-run …` | 显示资源、运行参数和未满足条件，不提交 |
| `submit RUN_ID --authorize …` | 用户明确批准后，远端 `sbatch --parsable` 提交 |
| `reconcile INTENT` | 核查不确定提交的远端回执与 Slurm 记录，不再次提交 |
| `status JOB_ID` | 一次性查询队列与记账状态，不持续高频轮询 |
| `logs JOB_ID --lines 80` | 读取有限日志尾部 |
| `fetch JOB_ID` | 将日志、指标、小结果取回独立本地目录 |
| `cancel JOB_ID --authorize` | 仅取消明确授权的那个作业 |

资源参数必须显式包含 account、partition、QoS、GPU 类型和数量、
CPU、主机内存与时限。当前工具的计算入口限定为 smoke；不能用
任意 shell 命令把首次测试扩展为训练。`--authorize` 是调用方对
本次明确批准的标记。本任务的一次首次 smoke 已经获准，可以在检查和
提交预览通过后使用；既往 G0 或其它平台的授权不能代替其它 Expanse
作业的授权。取消仍需要针对该作业的明确授权。
干运行与真正提交共用同一个 `sbatch_argv` 生成器，完整参数保存在
`.sdsc/submit-plan.json`，节点脚本为 `tools/sdsc_job.sh`。缺少运行时
或结果位置时，预览显示明确的 `UNCONFIRMED` 占位和 blockers；
它们不能用于真实提交。`--no-requeue` 禁止此微型测试自动重新执行。

提交前保存 intent 与稳定的提交标识，远端使用 `sbatch --parsable`
取得真实 job ID。提交记录绑定 job ID、run ID、快照哈希、精确资源、
运行时、结果路径与脚本。Quest 控制信息保存在 `.sdsc/`，该目录被
Git 和源码同步排除。只把有效回执作为已确认提交，不能自行
编造或猜测 job ID。

SSH 中断、回执丢失或返回格式异常时，标记为提交结果未知，先使用
`tools/sdsc reconcile INTENT`，通过相同标识核对远端持久回执、
`squeue` 和 `sacct`。**禁止盲目重试**。
即使暂时查询为空，也可能存在记账延迟；不能把空结果当作未提交。
只有确定原 intent 的状态后才能继续；不要换一个 run ID 掩盖未知
提交状态。取消操作同样不自动重试，也不取消所有用户作业。
每个 run ID 只能获得一个提交 intent；远端原子 claim 防止同时启动
两个提交进程造成重复作业。本地存在结果未知的 intent 时，也会阻止
为其它 run ID 提交，直到通过核查解决不确定性。

## 状态、结果与回传

`squeue` 中消失仅表示不再显示在活动队列。最终判定结合 `sacct`
中的 State、ExitCode、作业步骤，以及与 run ID／代码哈希一致的
实际结果。空记账、UNKNOWN、只有 exit 0 或只有日志中的“完成”
都不足以证明测试成功。

首次 smoke 的成功证据应覆盖：

- 实际运行的快照身份与上传清单一致；
- Slurm 分配与可见设备一致，确为一张 H100；
- 微型 GPU 运算输出有限且通过预期数值检查；
- 结果在计算节点写入指定持久位置后可再次读取；
- 作业 terminal State / ExitCode 可核对；
- Quest 回传结果保留作业身份与内容完整性。

`fetch` 只写 `.sdsc/fetched/<job_id>/`，不反向覆盖 Quest 源码，也不下载大型
checkpoint。日志读取和回传均有大小边界。失败后的日志和状态仍有
诊断价值，但不能成为成功标记；保留证据后再决定后续修复。

## 新会话操作顺序

Teacher 自主修复的当前入口以 `docs/refactor/current_handoff.md` 为准。
单卡适配预检 `54493015` 已完成执行验证；这不代表 teacher 质量验收通过。
新的 `qwen3-v2-teacher-fit-preflight` 与 `qwen3-v2-teacher-fit` 使用
4 H100、24 CPU、192 GiB，account `nwu181`，partition `nairr-gpu`，
QoS `nairr-gpu-normal`。前者最多一小时；后者的实际时间根据预检测量确定，
不可把 CLI 的 24 小时上限当作默认申请。使用独立的已验证运行时
`/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-teacher-adapt-peft0171-v1/bin/python3.12`。
两者均须 `--teacher-job-id 54493015`；完整训练还须显式指定本次成功的
四卡预检 `--preflight-job-id`，并与该预检匹配实际执行代码哈希。
提交器在创建 intent 前核查现有 GPU 作业，最多同时分配四张 GPU。

四卡预检固定 512 个独立训练样本、8 次 global-64 更新；完整适配固定
8,192 个训练样本、512 个独立开发样本、4 轮共 512 次更新。
在 128/256/384/512 步评估真实合并稠密模型，选第一个通过全部八项开发
指标的检查点，同时完成全部预定更新。新的 teacher-only 256-token
评估预算是明确的协议变更；学生原 128-token 设置保持不变。
原始阈值、完整覆盖和独立正式验收仍必须满足，不能把训练 loss 或
`teacher-fit.json` 的执行 `passed` 当作正式 teacher 成功。
权重留在 SDSC 持久存储；`fetch` 只取小报告和日志，超过上限的可选
报告明确列入 `skipped`，不会截断 JSON 或下载权重。完整科学定义见
`docs/refactor/sdsc_teacher_adaptation_20260928.md`。

以下操作始终在 Quest 项目终端执行。先读本仓库 `AGENTS.md`、
`docs/refactor/current_handoff.md` 和本节，检查 Git 工作区、
`.sdsc/supervision/*/state.json` 及 `.sdsc/submissions/` 已有记录。
**切换会话不重置单次作业授权**：
已有 job ID 时直接继续第 6 步；有 `sending`／`unknown` intent 时先
执行第 7 步，不能通过新快照或新会话绕过它。

下面第 2–5 步保留首次 smoke 的完整操作模板，**当前正式训练不要照抄
提交 smoke**。正式训练使用上文固定 Conda 路径、项目 Lustre 结果根及
对应任务资源；校准还需已核验的 provenance 和两个上游 job ID。
先读当前 supervision plan/state，再决定是否需要新提交。原 smoke 的
单次授权已经使用，后续正式训练授权则按本文开头和 AGENTS.md 执行。

1. 检查当前主机的已有连接。`tools/sdsc check` 会先执行
   `ssh -O check`，每次自行计算
   `$HOME/.ssh/cm/sdsc-$(hostname -s)`；不依赖认证终端的变量。
   缺少 master 时停止，由用户按照[认证与连接检查](#认证与连接检查)
   在**同一台 Quest 主机**手动认证，然后再继续，不循环尝试。

2. 检查现场运行时、account/QoS 和目标路径。以下 Bash 数组仅用于
   本段命令复用，各参数均在此显式赋值；工具内部不依赖这些变量。

   ```bash
   cd /gpfs/projects/p32737/del6500_home/OPD
   sdsc_runtime_args=(
     --python /usr/bin/python3.11
     --container-runtime /cm/local/apps/singularitypro/4.1/bin/singularity
     --container-image /expanse/projects/qstore/installs/containers/singularity/Expanse-Air/pytorch/pytorch-nvcr-25.03.sif
     --container-python /usr/bin/python
     --result-root /home/zgao12/quest-runs/OPD/smoke-results
   )
   tools/sdsc check "${sdsc_runtime_args[@]}"
   ```

   核对远端是 `zgao12`，容器检查成功、实际参数仍有效。某项查询超时
   只表示该项当前未验证，不能写成通过；连接正常也不代表 GPU 已验证。

3. 审阅源码清单和大小，再上传：

   ```bash
   tools/sdsc sync --dry-run
   tools/sdsc sync
   ```

   文件修改后必须重新 dry-run。成功上传回执的 `run_id` 与最近匹配
   预览相同；没有成功回执时，不把预览当成已部署。首次上传会创建
   `smoke-results` 小结果目录，随后重跑第 2 步的 `check` 检查权限。

4. 将下列占位值替换为**成功上传回执**中的真实 ID，审阅准确的
   `sbatch_argv` 和 blockers：

   ```bash
   sdsc_run_id='REPLACE_WITH_SUCCESSFUL_SYNC_RUN_ID'
   sdsc_smoke_args=(
     "$sdsc_run_id" --task gpu-smoke
     --account nwu181 --partition nairr-gpu-shared
     --qos nairr-gpu-shared-normal
     --gpu-type h100 --gpus 1 --cpus 4 --mem-gib 16 --time 00:05:00
     "${sdsc_runtime_args[@]}"
     --storage-confirmed
   )
   tools/sdsc submit "${sdsc_smoke_args[@]}" --dry-run
   ```

   `--storage-confirmed` 仅在已确认 HOME 的持久属性、目标路径及登录节点
   权限后使用；它不能代替作业内 H100 挂载／读写验证。镜像元数据变化、
   路径缺失或其它 blocker 未解决时先停下，不修改记录绕过检查。

5. 以下是提交命令模板。必须有覆盖当前具体新作业的用户授权，确认
   该授权尚未用于已有提交、预览无 blocker 后，才执行**一次**：

   ```bash
   tools/sdsc submit "${sdsc_smoke_args[@]}" --authorize
   ```

   保存输出中的 intent 和真实 job ID。此命令通过 SSH 调用 SDSC
   `sbatch --parsable`；不要在 Quest 手动调用 Slurm。此授权不包括
   失败后的第二次提交、扩大资源、正式训练、G0 或无限重试。

6. 按需查询并回传。把 `JOB_ID` 替换为真实数字 ID：

   ```bash
   tools/sdsc status JOB_ID
   tools/sdsc logs JOB_ID --lines 80
   tools/sdsc fetch JOB_ID
   ```

   查看 `sacct` 的终态、ExitCode 和工具的结果验证；不能把队列消失、
   提交回执或测试替身当成成功。回传只写 `.sdsc/fetched/`。无终态时
   稍后按需再查，不启动高频轮询或常驻服务。若作业失败，保留日志、
   结果和诊断，另行确认下一次计算的授权。

7. 提交断线或无有效回执时，用原 intent 核查：

   ```bash
   tools/sdsc reconcile INTENT
   ```

   恢复 job ID 后继续第 6 步；仍为 UNKNOWN 就保持未知，不重复提交。
   需要取消时，先取得针对该作业的明确授权，才使用
   `tools/sdsc cancel JOB_ID --authorize`。不取消其它作业。

## 首次真实 H100 测试结果

2026-09-17（Quest 当地时间）在用户明确授权后完成一次测试，未重试。

| 项目 | 实测值 |
| --- | --- |
| Job ID / intent | `54345483` / `ed6b17d1d1fd45a58c4fc206766d6324` |
| Run ID | `20260918T013816Z-1e6fb5739378-e1ff0d7a` |
| 提交时间 | `2026-09-18T01:39:22Z` |
| 资源 | 1 H100、4 CPU、16 GiB；申请 5 分钟，Slurm 记录运行 16 秒 |
| 终态 | 作业、batch、extern 均 `COMPLETED`、`ExitCode=0:0`；工具 `success=true` |
| 节点 / GPU | `exp-19-04` / `NVIDIA H100 80GB HBM3`，可见设备仅一张 |
| GPU 运行时 | 实际导入 Torch `2.7.0a0+7c8ec84dab.nv25.03`，CUDA 12.8 |
| 微型计算 | 128×128 矩阵乘积，所有值有限，与 CPU 参考一致 |
| 节点工作盘 | `/scratch/zgao12/job_54345483/`，本地 ext4；源码暂存前后校验通过 |
| 持久化 | HOME NFS 专用小结果目录，计算节点写入、回读校验通过 |
| 回传 | 六个日志／JSON 文件，共 9,216 字节，Quest 端再次校验哈希 |

源码 SHA-256 为
`1e6fb5739378f917b15edd54538519a1480bedd0945db738d101426d16f554a8`；
结果 SHA-256 为
`3462d93389b2e8ff3dec45afae1fd0209f12147a9df8b281dab6624b668bf817`。
本地提交回执在 `.sdsc/submissions/ed6b17d1d1fd45a58c4fc206766d6324.json`，
最终记账与校验在 `.sdsc/job-54345483-final-status.json`，回传目录为
`.sdsc/fetched/54345483/fetch-53tjr6t4/`。远端结果位于
`/home/zgao12/quest-runs/OPD/smoke-results/20260918T013816Z-1e6fb5739378-e1ff0d7a/ed6b17d1d1fd45a58c4fc206766d6324/`。
独立只读复核也确认提交／结果身份、文件哈希及日志一致。

这证明本次节点上的源码部署、本地执行、现有容器 GPU 计算、小结果
持久化和回传链路可用。它没有验证 Lustre、正式数据与模型缓存、训练
checkpoint 存储、NCCL/FSDP 或 G0；HOME 仍仅用于小元数据。正式实验
需要另行准备匹配项目版本锁的运行时和输入／大结果存储。报告中的
`peak_host_rss_kib` 仅是宿主控制进程，不代表容器或整个作业内存峰值。
现有快照保持不变；此次测试之后的文档更新没有覆盖部署代码。

## 第一阶段实测记录

2026-09-17（Quest 当地时间；最新记录 UTC 为 2026-09-18 01:32:58），
在 `quser44` 复用用户手动认证的 master 成功。实际 ControlPath 为
`/home/del6500/.ssh/cm/sdsc-quser44`，远端身份为 `zgao12@login01`。
`sbatch`、`squeue`、`sacct`、`scancel`、`findmnt` 等必要命令均存在。
在这一检查记录产生时，没有运行登录节点 GPU 运算、安装软件、下载
模型、上传快照或提交作业。该记录描述首阶段检查，不代表后来执行状态；
随后获准的单次 smoke 以 handoff、提交回执及 Slurm/结果记录为准。

现场 account 关联与分区 `AllowQos` 的交集仅包含
`nairr-gpu-shared-normal`；最初提供的 `nairr-gpu-shared` 不是该交集的
有效 QoS。工具和下面的命令已使用实测值。`expanse-client user -r
expanse_nairr_gpu` 返回 `STATE=allow`、`PROJECT=nwu181` 和
`AVAILABLE=10000`；这里保留工具的原始数值，不推定其计费单位。

HOME 和选定镜像所在的 qstore 在**登录节点**显示为 NFS。该次检查时
目标项目根及 `smoke-results` 尚未创建；HOME 可写。H100 节点上镜像／HOME 的可达性、
节点本地工作盘以及小结果持久化和回传，仍须由随后获准的 smoke 实测。
smoke 不需要外部数据或模型；正式实验的数据、模型缓存和大结果持久
存储仍未确定，不能根据此次登录节点检查视为已经就绪。

本地记录位于被忽略的 `.sdsc/`：`check.json` 为最新工具检查，
`discovery.json`、`runtime-followup.json` 和 `air-container.json` 为
环境发现证据；`last-preview.json` 为源码清单，`submit-plan.json`
为完整 `sbatch_argv`。这些记录可能随以后检查更新，不能代替现场状态。

以下是第一阶段使用的**本地干运行**。将 `RUN_ID` 替换为
`sync --dry-run` 输出的 ID；不需要上传即可生成提交预览：

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
tools/sdsc sync --dry-run
tools/sdsc submit RUN_ID \
  --task gpu-smoke \
  --account nwu181 \
  --partition nairr-gpu-shared \
  --qos nairr-gpu-shared-normal \
  --gpu-type h100 --gpus 1 --cpus 4 --mem-gib 16 --time 00:05:00 \
  --python /usr/bin/python3.11 \
  --container-runtime /cm/local/apps/singularitypro/4.1/bin/singularity \
  --container-image /expanse/projects/qstore/installs/containers/singularity/Expanse-Air/pytorch/pytorch-nvcr-25.03.sif \
  --container-python /usr/bin/python \
  --result-root /home/zgao12/quest-runs/OPD/smoke-results \
  --dry-run
```

干运行在 Quest 本地生成，不调用远端 `sbatch`，资源精确为
**1 H100 / 4 CPU / 16 GiB / 00:05:00**。当时预览保留两个阻塞项：
快照尚未上传、结果持久位置尚未最终确认；容器与宿主 Python 的路径
已有现场证据。第一阶段到此结束。

用户随后授权的首次 smoke 已执行成功，结果见上节。未来操作采用
[新会话操作顺序](#新会话操作顺序)，先检查已有提交记录；本次授权
已经使用，不将其扩展为后续作业授权。

本地验证命令：

```bash
/usr/bin/python3.12 -I -B -m unittest discover -s tests/sdsc -p 'test_*.py' -v
bash -n tools/sdsc tools/sdsc_job.sh
```

本次 90 项本地测试、Ruff 与 Shell 语法检查通过，包括真实归档与
校验器接口、未提交内容、凭证路径排除、符号链接拒绝、并发去重、
丢失回执、容器参数及身份绑定、终态／产物联合判断，以及计算脚本的
持久化失败路径。所有调度与 GPU 调用均为测试替身；这些结果不代表
已在 Expanse GPU 上运行成功。变更留在当前工作区，未创建 Git 提交。

官方指南将 `expanse-client user -r RESOURCE` 作为额度查询方式，
并将 batch account 与查询 resource 分开；本项目分别使用
`expanse_nairr_gpu` 和 `nwu181`。指南也限制登录节点承担高负载计算、
大规模传输或工作流服务。[SDSC 账户与登录说明](https://www.sdsc.edu/systems/expanse/user_guide.html)
