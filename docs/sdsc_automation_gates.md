# SDSC G0 → seed-42 pilot 自动接续与验收边界

审查日期：2026-09-18 UTC。本文是实现与科学验收边界的工程记录，不是 G0
通过证明、H100 执行类证书或新的实验结果。源码行号对应本次审查时的工作区，
后续改动时应以函数及实际源码为准。科学 checkout 的真实 HEAD 为
`0215c356355b29b5e2b407978a207db2156719e1`；未提交的 SDSC 工具另由源码快照绑定。

用户已经授权 G0 后接 seed-42 pilot。必要的实现、测试和独立审查可继续完成，
不因自动化阶段划分再次索取相同授权。完整三种子 factorial 和 Gemma 不在该范围。
自动化负责满足条件后接续、失败后停止，不能修改判据以制造通过。

后继实现分成固定计划/SSH 提交、节点本地暂存与持久化、G0 两次恢复与终验、
四卡预检、完整八方法 pilot。入口、资源上限和恢复命令见
[完整自动接续](sdsc_workflow.md#完整自动接续)。下文保留原阶段/科学门槛
审计，不能把旧代码中的问题当作新适配器已取得的 GPU 通过证据。

## 1. 当前真实证据与实现状态

以下运行状态来自已保存的本地证据，不是本次文档审查新发起的远端查询。

| 对象 | 已实现 | 已取得的真实证据 | 尚不能据此宣称 |
| --- | --- | --- | --- |
| SDSC SSH、源码快照、Slurm receipt/status、有限 fetch | 是 | 基础 smoke `54345483` 完成；Lustre 节点读写 smoke `54345521` 完成 | 模型训练、G0、pilot 通过 |
| 两 H100 模型/分布式 preflight | 是 | `54345604`：allocation/batch/extern 全部 `COMPLETED / 0:0`；真实模型、NCCL、一个 global-64 optimizer window、完整状态保存恢复及持久化读回通过 | 多步 resume、四卡能力、online/soft/GRPO 方法或 circuit 路径全部通过 |
| 教师数据任务 | 是 | 已提交真实 job `54345715`；本地 supervisor 在 `2026-09-18T02:51:29Z` 保存的观察仍为 `RUNNING`，阶段为 `waiting_teacher` | 2048 次生成完成、256 prompts 全覆盖或 teacher readiness 通过 |
| 两 H100 canonical-SFT calibration | 是 | CPU fixture、真实 clean-checkout 配置绑定及真实 `train --dry-run` 已验证；本次状态快照中没有 calibration job ID | 正式训练已经完成、完整 G0 或 resume 验收通过 |
| teacher → calibration 有限 supervisor | 是且已启动 | 固定计划/状态位于 `.sdsc/supervision/teacher-54345715-to-calibration/` | 自动执行剩余 G0 或 pilot |
| runtime checkpoint/science 配置绑定 | 已实现并接入新 SDSC finalizer | `tools/sdsc_science_binding.py` 验证原科学投影与实际 checkpoint 字节 | 旧证书已变为 H100 证书 |
| 完整 SDSC G0 与 seed-42 pilot 自动执行 | 工具与独立审查已完成 | 新 `tools/sdsc_pipeline`，真实启用状态见 `.sdsc/supervision/teacher54345715-g0-pilot-v1/`；CPU 测试不代替 GPU | 实际 G0/pilot 已通过或所有未来作业一定成功 |

两卡结果的保存证据为
`.sdsc/preflight-54345604-final-status.json`，其中 `success=true`、
`result.verified=true`，报告原始 SHA-256 为
`53524a9263bc21983073b5859f7332500371330cc70136fc48f9e78481cf3b43`。
可变状态继续以实际 receipt、Slurm accounting 和产物语义验证为准。

**保持当前 supervisor 及其固定控制文件不变，不修改、不重启、不重复 arm。**
其实现只处理 teacher → calibration，见
[`tools/sdsc_supervise.py:174`](../tools/sdsc_supervise.py#L174)。后继流程应有独立的
不可变计划、上游证明、唯一提交 claim 和 receipt reconciliation；仅生成候选目录或
将 watcher 停在 `ready` 不算完成自动接续。

## 2. 完整 G0 阶段 DAG

权威阶段清单是
[`qwen3-v2-g0-handler.py::_stage_plan:2128`](../scripts/server_scheduler/qwen3-v2-g0-handler.py#L2128)。
下面保留其 19 个阶段及科学含义。表中“可复用”都以原始文件哈希、真实生产者身份、
已完成作业及相应科学 validator 通过为前提，不能仅判断目录存在。

| # | 科学 CLI / 阶段 | 必要前置与输出 | SDSC 接续工作 |
| --- | --- | --- | --- |
| 1 | `build_splits` | 完整七个 split、共 144000 examples → `dataset/` | 复用成功 teacher 任务的完整 family，不重新生成 |
| 2 | `export_initial_checkpoint` | 固定 Qwen3-1.7B revision → `initial_checkpoint.pt` | 复用成功 calibration 实际导出文件及 SHA，不重复导出 |
| 3 | `build_teacher_demos` | 固定 256 个 manifest-order prompts、每个 8 candidates、seed 31415 → 完整 ledger/view/store | 复用成功 teacher 任务；完整 coverage/verifier gate 不变 |
| 4 | `audit_label_leakage` | family validation split → `label_leakage.json` | 科学 CLI 已有；SDSC 阶段与 receipt 尚需实现 |
| 5 | `evaluate_teacher_readiness` | 固定 teacher、validation 128 examples、top-k 128 → `teacher_readiness.json` | 必须实际执行；teacher-demo coverage 不能代替此门槛 |
| 6 | `train` / `calibration_sft` | teacher store + initial checkpoint → uninterrupted reference、metrics、step20 与终态 checkpoint | 复用已实现 calibration 任务；SDSC 目录名为 `canonical_sft` |
| 7 | `score_probe_candidates` | 初始与 reference 终态 checkpoint、真实 run manifest、family；每 circuit split 128 个有序 pair groups、task validation 128 → `probe_scores.json` | 科学 CLI 可直接消费 `canonical_sft` manifest；先严格验证完整训练产物 |
| 8 | `build_probe_cohorts` | dataset family + probe scores → `probes/manifest.json` 等 | 科学 CLI 已有；保留 pair 顺序、split 隔离及 eligibility ancestry |
| 9 | `build_rollout_bank` | 固定初始 student、配置 prompt population → `rollout_bank/` | 新 SDSC 阶段，保持原 RNG/生成语义 |
| 10 | `score_teacher` | 上一步 bank、固定 teacher、`experiment=offline_soft` → `teacher_scores/` | 实际 top-k scoring、文件完整性及混合 reward 检查 |
| 11 | `evaluate_anti_shortcut` | initial checkpoint + 固定 suite → `anti_shortcut.json` | 实际生成/验证；不能用训练 loss 或普通 accuracy 替代 |
| 12 | `discover_circuit --stage final_answer` | base-capable discovery cohorts、initial checkpoint、固定 MIB → final-answer circuit 与 compatibility | 准备 MIB/H100 路径并实际运行 |
| 13 | `evaluate_circuit`，final answer | 上一步 circuit、held-out cohorts、initial checkpoint → exact patching | 保留 identity/full-corruption/random-control/calibration 检查 |
| 14 | `discover_circuit --stage first_rule_selection` | 独立 process-stage targets、同样模型/数据绑定 → process circuit | 与 final-answer 阶段分开，不能复用同一个 stage manifest |
| 15 | `evaluate_circuit`，process | process circuit、held-out cohorts → exact patching | 同样保留完整验证 |
| 16 | `train --resume`，replica a | reference 非终态 step20 + 完整同世界尺寸状态 → `resume-a/` | 新的真实两卡续训；先补跨作业路径映射 |
| 17 | `train --resume`，replica b | 同一个未经修改的 step20 → `resume-b/` | 独立第二次真实续训，不能复用 replica a 的结果 |
| 18 | `compare_distributed_resume` | reference、a、b 的完整产物 → `distributed_resume.json` | 保留逐项终态比较；适配原目录名和显式路径映射 |
| 19 | `finalize_g0` + semantic replay + publication | 上述完整证据、有效科学/执行绑定 → G0/readiness/产物清单 | 新 H100 完成边界尚需实现与独立审查；旧入口不可冒用 |

依赖关系可组织为：

```text
完整 dataset ──→ label leakage ──────────────────────────────────────┐
      ├──────→ teacher readiness ────────────────────────────────────┤
      └─→ teacher demos + initial checkpoint → calibration reference │
                         │                       ├─→ probe scores     │
                         │                       │    → frozen cohorts│
                         │                       │    → 两阶段 circuits
                         │                       │    → exact patching│
                         │                       └─→ resume a / b     │
                         │                            → 严格比较      │
                         ├─→ rollout bank → teacher scores ──────────┤
                         └─→ anti-shortcut ──────────────────────────┤
                                            全部证据 + H100绑定 ────→ G0
```

这些分支可以独立准备，但顺序优化不能改变任何 RNG、数据或筛选定义。
优先完成 teacher readiness、probe/base accuracy、anti-shortcut 等可提前发现科学失败的
分支，再启动昂贵 circuit/resume，是减少无效 GPU 消耗的候选执行顺序。
不把原 handler 的单次大作业重新完整跑一遍。

## 3. 可复用产物与路径重定位

成功 calibration 的最小保留集合是：完整 dataset、完整 teacher store、
initial checkpoint、`canonical_sft/` 的 manifest/resolved config/ConfigBinding/metrics/
update evidence、终态 checkpoint、**非终态 step20 checkpoint 及所有 `.accelerate` 文件**。
checkpoint 内部和外部 inventory 的哈希都须验证；保留原生产者 HEAD、run/job ID、
源码快照、runtime 和 publication receipt。

[`validate_factorial_run_artifacts:373`](../src/posttrain_circuits/cli/factorial_run_validation.py#L373)
已经支持“配置内 source workspace → 实际提取 workspace”的只读产物验证。
[`score_probe_candidates:111`](../src/posttrain_circuits/cli/score_probe_candidates.py#L111)
通过显式 manifest/checkpoint 参数调用它，所以不要求把 `canonical_sft` 重命名为
`calibration`。原 resolved config 和生产者身份必须保留。

真实 resume 尚有两处不同的约束：

- [`compare_distributed_resume:204`](../src/posttrain_circuits/cli/compare_distributed_resume.py#L204)
  固定读取 `workspace/calibration`，并要求 checkpoint 中完整 resolved config 与当前比较配置相同。
- [`FactorialTrainer.resume:2445`](../src/posttrain_circuits/learning/training/factorial_trainer.py#L2445)
  校验实际路径为规范普通文件；在加载任何训练状态前验证 world size、cursor、token 和
  optimizer-boundary 元数据。其
  [`accelerate_state_dir` 检查:2501](../src/posttrain_circuits/learning/training/factorial_trainer.py#L2501)
  要求保存的绝对路径等于当前 checkpoint 的兄弟目录；新 Slurm 作业的 scratch 根通常不同。

因此“复制目录后原样运行旧 resume CLI”不成立。最小正确适配应显式绑定原 workspace、
实际 node-local workspace、原 reference role/path 与所有内容 SHA；只将读取位置映射到
已验证的副本，原 checkpoint、manifest 和 resolved config 字节不变。保留旧路径的逻辑
含义及 no-symlink/containment 检查，并为路径逃逸、changed-world、半窗口、缺文件、
哈希失配设置拒绝测试。**不能重写旧 checkpoint 的路径字符串后重新散列，或将
历史产物改标成新的 HEAD。** 此适配已由 `tools/sdsc_resume.py` 实现并独立审查；原路径仅通过私有 bind 恢复，真实 GPU 结果仍须运行后验证。

若修改科学 `src/` 导致 producer/consumer HEAD 不同，必须有真实的跨版本兼容审查与
明确 lineage；不能隐去旧生产者身份。优先将控制/存储适配与科学 checkout 分开绑定，
避免无科学变化却迫使 dataset、teacher 和 reference 全部重算。

## 4. 完整 G0 科学与运行时门槛

[`G0_CHECK_NAMES:73`](../src/posttrain_circuits/cli/finalize_g0.py#L73) 定义 28 项检查。
新 SDSC 完成路径必须保留完整集合和实际重放，不能只检查 `passed=true`：

| 类别 | 必须保留的原检查 |
| --- | --- |
| 数据、teacher、能力 | `label_leakage`, `teacher_correctness`, `teacher_topk_mass`, `fixed_bank_mixed_rewards`, `base_task_accuracy`, `calibration_anchor_improves_accuracy`, `anti_shortcut` |
| probes 与身份 | `base_capable_probes`, `probe_scoring_binding`, `split_probe_isolation`, `artifact_reconstruction`, `distinct_stage_manifests`, `qwen3_protocol_bindings`, `protocol_amendment` |
| circuit 实测 | `hf_transformerlens_gqa_parity`, `qwen3_qk_norm_hook_semantics`, `bootstrap_stability`, `final_stage_eap_beats_matched_random`, `process_stage_eap_beats_matched_random`, `identity_sanity`, `full_corruption_sanity`, `attribution_exact_calibration` |
| 分布式与绑定 | `batch_token_invariants`, `distributed_checkpoint_resume`, `distributed_resume_fsdp_strategy`, `execution_safety_descriptor`, `execution_safety_certification`, `execution_science_protocol` |

具体科学判定见
[`finalize_g0:1112`](../src/posttrain_circuits/cli/finalize_g0.py#L1112)：base accuracy
至少配置阈值 0.10；calibration accuracy 严格改善；bank 必须既有正 reward 又有负 reward；
teacher retained top-k mass 最小值至少 0.90；两个 base-capable subsets 非空；
两阶段 bootstrap Spearman 至少 0.50；selected-vs-matched-random CPR margin 严格大于 0；
attribution/exact Spearman 至少 0.20、bootstrap 下界至少 -0.05。
阈值仍由原配置和 validator 定义，不从本文重新生成或放宽。

Teacher readiness 独立要求 answer accuracy ≥0.90、exact proof ≥0.85、
first-rule/intermediate top1 ≥0.80、top-k target coverage ≥0.90、corrupted-prefix
recovery ≥0.70 及非负 causal shift，见
[`TeacherReadinessThresholds:16`](../src/posttrain_circuits/learning/teacher/evaluation.py#L16)。
生成 store 覆盖全部 prompts 并不能证明这些指标。

真实 resume 必须从非终态 optimizer-boundary step20 出发；文件存在并不等于可恢复。
两个 replica 要与 uninterrupted reference 的 model/optimizer/scheduler/RNG 等 core
runtime hashes 完全一致，global step、token state、终止原因、world size 和 FSDP
策略一致；ancestry 必须绑定同一个 step20 SHA，loss 误差不超过固定 `1e-6`。完整逻辑见
[`compare_distributed_resume:240`](../src/posttrain_circuits/cli/compare_distributed_resume.py#L240)
和 [`_build_comparison_payload:341`](../src/posttrain_circuits/cli/compare_distributed_resume.py#L341)。
现有单窗口保存/扰动/恢复的真实 preflight 不能代替两次多步续训。

### H100 与 MIB 的额外真实验证

G0 继续采用两 H100、24 CPU、192 GiB、固定 Python 3.12.13 及 G0 dependency lock，
保持 global batch 64、microbatch 上限 4、最大模型输入 1536、120 optimizer steps /
2000000 全局非 padding tokens、W=2 FULL_SHARD。固定模型 revision 和完整 offline cache
沿用已验证输入。每个新作业仍检查实际 GPU、mount、cgroup ancestor 有效上限与 headroom。

MIB 是额外外部源码依赖，不能从“19 个 Python 包已安装”推断已具备。要求 checkout
`b759df34433c9e31043ba9e02908ce0bf20e894f`、真实干净 Git 身份、gitlink 固定的递归
子模块和 `run_attribution.py`；检查依据是
[`MIBEAPIGBackend:142`](../src/posttrain_circuits/causal_circuits/discovery/backends/mib_eap_ig.py#L142)
及原 runtime preparation 的
[`submodule` 校验:183](../scripts/server_scheduler/prepare-qwen3-v2-g0-runtime.py#L183)。
复用依赖准备逻辑，不执行原脚本中的 `/scr/del6500/OPD` 部署路径。

两阶段必须在 H100 上实际验证 HF/TransformerLens logits parity、GQA K/V 映射、
Qwen3 Q/K norm hook 语义、EAP-IG 与 exact patching。新的 circuit/teacher-scoring/
anti-shortcut 路径不由 canonical-SFT 的显存证据覆盖；尤其 anti-shortcut 的现有输入
包络可超过 1536，不能错误声称都在训练包络内。先以实际配置验证资源，再提交对应阶段。
不因旧执行类支持 1/2/3/4 就自动重复四个 pilot；H100 新证据只声明实际验证的范围。

### 旧 finalizer 的两个必须显式处理的边界

[`finalize_g0.main:947`](../src/posttrain_circuits/cli/finalize_g0.py#L947) 要求
ServerScheduler handler-owned `ScientificInvocationContext`，其 certificate/class
身份不是 H100 Slurm 权威。不得捏造旧 context、借用 Blackwell 证书或删除对应检查。
H100 的执行身份、真实分配、runtime、适配器及有效证据需有独立接受的完成合同；
最后仍须重放全套科学判定，并验证持久化产物和真实 Slurm 终态。

另一个已 CPU 复现的晚期问题是：原 handler 给最终阶段传入真实
`production_safety.initial_checkpoint_hash`，冻结科学配置该字段为空，而
[`_load_execution_science_protocol:306`](../src/posttrain_circuits/cli/finalize_g0.py#L306)
仍比较保留此字段的 canonical science hash。原流程会在末阶段拒绝配置。
不能全局忽略该字段、篡改已接受协议，或只对 finalizer 清空 hash。

候选 [`sdsc_science_binding.py`](../tools/sdsc_science_binding.py) 将“冻结 base 配置”与
“经过独立 SHA 校验的真实 initial checkpoint 所产生的 runtime 配置”分别绑定，并只允许
这个已验证的派生字段变化。它没有修改旧 hash 算法或 finalizer，不生成 G0/执行证书；
完成路径的集成与独立审查仍是必需工作。

该候选的真实 clean science checkout 完整 CLI 验证已通过；
base 仍为已接受的 `6c3942…4662657`，改变 seed 到 43 被拒绝。CLI 使用小型字节
fixture 验证实际哈希，并未把它视为模型 checkpoint 或 GPU 证据。扩展后的 SDSC
完整测试曾通过 261 项。独立审查随后发现并修复 producer 布尔值/整数比较问题，
增加一项回归，受影响的 18 项 binding 测试及 Ruff 再次通过，独立复审无阻断。
实际字节 fixture 报告保存在 `.sdsc/automation-audit/science-binding-cli-byte-fixture.json`。
该工具现已接入新 `sdsc_finalize_g0.py` 与独立 `sdsc_pipeline` 后继；原旧 finalizer 和当前 teacher/calibration 控制文件未改。

## 5. G0 后的 seed-42 pilot

Pilot 拟使用四 H100，以保留原四 GPU 的 batch/恢复语义；实际分区资源、
四卡预检与运行时仍须验证，不能把当前两卡证据改标成四卡。
[`qwen3_v2_core.yaml:10`](../configs/pilot/qwen3_v2_core.yaml#L10) 固定 seed42、
4096 prompts、每方法最多 120 steps / 2M tokens、每 20 steps checkpoint/evaluation、
validation128 和 full-parameter training。

固定矩阵为六个 factorial cells：`offline_hard`, `online_hard`, `offline_soft`,
`online_soft_opd`, `offline_verified_replay`, `online_verified_replay`，以及两个 anchors：
`canonical_sft`, `canonical_grpo`。后续包括初始 2 stages × 2 cohorts = 4 个
circuit/exact 组合，最终 8 methods × 2 stages × 2 cohorts = 32 个组合、
local-fork horizons 1/5/20、SFT step20 的两次 resume 和 32 个 dynamics。
矩阵解析依据是
[`resolve_pilot_circuit_scope:18`](../src/posttrain_circuits/causal_circuits/discovery/pilot_scope.py#L18)，
local-fork 原入口见
[`pilot_local_fork.slurm:36`](../scripts/slurm/pilot_local_fork.slurm#L36)。

原 SFT/factorial physical batch 是 `4 × accumulation4 × world4 = global64`，
[`fsdp.yaml:5`](../configs/accelerate/fsdp.yaml#L5) 固定四进程；直接用两卡变成 global32。
[`finalize_pilot:674`](../src/posttrain_circuits/cli/finalize_pilot.py#L674) 还要求 resume
world size 为 4。Online generation、soft teacher、verified replay 与 GRPO 都有各自的
资源和科学语义，canonical-SFT 的两卡预检不能替它们完成验证。
GRPO 还要求 [`trl==0.22.2`](../pyproject.toml#L38)，当前 G0 的 19 包锁不包含该依赖；
应固定和验证 pilot runtime，避免就地变更正在运行或已绑定的 G0 runtime。

[`prepare_pilot:26`](../src/posttrain_circuits/cli/prepare_pilot.py#L26) 校验 G0 自哈希、
`passed`、科学 schema、精确 HEAD 和模型/prereg 等身份，但不会重放全部 28 个 G0 gates。
调用它之前，控制面必须已经完成完整 G0 semantic replay、产物清单验证和真实 Slurm
终态检查。Pilot 最终验收还检查所有方法的完整 artifact chains、共享 bank、frozen cohorts、
local fork、resume、circuit matrix 和 dynamics；
[`finalize_pilot:718`](../src/posttrain_circuits/cli/finalize_pilot.py#L718) 明确不以观察到的
效应方向作为通过门槛。

旧 [`submit_pilot.sh:69`](../scripts/production/submit_pilot.sh#L69) 使用本机 `sbatch`、
旧 artifact 布局和环境约定，不能从 Quest 直接调用。原
[`finalize_pilot_training::_terminal_tasks:107`](../src/posttrain_circuits/cli/finalize_pilot_training.py#L107)
要求真实同一 Slurm array 的八个任务 0..7。新控制面应支持并绑定真实 array evidence；
若选择八个独立 job，必须先完成新的语义验收适配，不能伪造 array accounting。
这些科学 CLI 可作为迁移组件，不等于旧生产/Slurm 脚本重新成为受支持入口。

## 6. 已实现的后继自动化契约

1. 让已启动的 teacher → calibration 流程按其固定计划继续；读取真实 receipts/status，
   不重启、不复用 intent/job ID，不因 SSH 回执不明确重新提交。
2. 为后继 G0 构建独立不可变 DAG，绑定 calibration 和 teacher 的完整产物与生产者，
   提交前完成 MIB/runtime、路径重定位、动态 checkpoint/science binding 的实现及独立审查。
3. 以真实小型 CPU fixtures 验证完整的 producer→consumer/schema/路径/哈希接口；
   完成前置判断后执行仍缺少的实际科学分支和两次 resume，复用已有 reference。
4. 保存每阶段真实 job/array ID、资源、code/runtime/input hashes 和输出路径；
   每一步只接受 accounting、持久化读回和语义 validator 同时通过。
   失败/TERM 保留诊断与完整稳定 checkpoint，不复制正在写的文件、不暗中恢复或重复提交。
5. 完整 G0 semantic replay 通过后才生成 pilot manifest，自动接续已授权的固定 seed42
   pilot；在四卡、各方法 runtime 和 array receipt 边界完善前，不把“已准备”说成“已启用”。
6. 有限 Quest supervisor 驱动固定后继任务和唯一 claims；SDSC 仅接收部署和 Slurm 作业。
   新流程的实现、审查及实际通过证据需要分别记录。任何科学门槛失败都停止依赖阶段，
   不自动降阈值、扩大实验范围或将失败产物升级成成功证据。

本次文档审查没有修改现有 supervisor/control、handoff、科学源码或历史产物，
没有 SSH、GPU 执行或作业提交。
