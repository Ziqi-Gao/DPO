# 诊断检查与正式训练自动接续

用户已授权：自动检查诊断作业，科学条件满足后自动推进正式训练，不再
逐阶段请求相同授权。入口是 `tools/sdsc_auto_continue`。它在 **Quest**
运行，消费现有只读观察器的结果，不安装 SDSC 服务，不重复轮询远端。

## 两层成功条件

对当前 v7 作业 **54485969**，自动接续重新核验观察器的计划、真实提交
回执、Slurm accounting、发布清单和报告哈希。只有执行/持久化成功，且
256 个候选覆盖全部 **32/32** 训练提示，才会触发一次后续 Codex 会话。
作业 `COMPLETED / 0:0` 或诊断报告 `passed=true` 单独不能触发训练。

后续会话也留在 Quest，执行固定的
[接续任务](sdsc_auto_continuation_task.md)：冻结 v7，验证预先确定的
`[128:256]` 补充样本组及原首 128 门槛，完成所需工具迁移和独立科学验收，
生成完整 256×8 教师数据并验证正式 readiness，运行同源两卡预检，再接续
calibration → 完整 G0 → 四卡预检 → seed-42 pilot。全部门槛保留，
任一失败停止。旧 v3 流程不会重新启动，三种子全因子和 Gemma 不在范围内。

这会自动执行必要的实现、审查和提交工作，但不能预先保证科学结果通过。
启动 Codex 进程不代表学生训练已经开始；训练状态必须有真实 Slurm job ID
和对应产物证据。

## 运行边界

- 每五分钟读取原观察器状态，最长十四天；状态失联超过二十分钟即停止。
- 固定实际 Quest 主机/仓库、job/run/源码身份、观察器计划、控制工具、
  接续任务文档和 Codex 可执行文件哈希；触发前变化即停止。
- 每个诊断 job 只有一个永久启动 claim。重复启动、新 flow 或进程退出
  不会自动重试，需先核查 claim、后继会话和所有提交回执。
- 触发前检查现有 SSH master。连接不可用时停止，不自动认证。
- Codex 使用本机已验证的 `exec --approve-for-me`，保留 workspace-write
  沙箱和自动审批审查。该 CLI 版本不允许同时传 `--sandbox`；不使用
  `danger-full-access`、审批绕过或忽略项目规则的参数。
- 一次后继 Codex 进程最多八小时，必要的长期 GPU 等待交给有限 Quest
  supervisor。超时只终止本次启动的 Codex 子进程，不取消 Slurm 作业。
- Codex 退出码不作为科学通过证据。最终检查真实 accounting、产物哈希、
  readiness/G0/pilot validator 和实际已启动的后继流程。

这是项目自己的有限 Quest 进程，不是应用内定时任务，也不是常驻服务。
官方 [Codex 非交互模式文档](https://learn.chatgpt.com/docs/non-interactive-mode)
说明了从脚本启动 `codex exec` 的接口；具体参数以本机 0.156.1 帮助和
`.sdsc/automation-check/` 中的真实握手记录为准。

## 状态与恢复

旧作业 54472139 因中断失败，其观察器已结束、自动接续已停止且未启动训练。
用户随后授权修复和重提；当前绑定的是新作业 54485969。保留旧 flow 的
计划、状态和失败证据，不能重启旧 flow。修复及资源边界见
[中断修复记录](refactor/sdsc_probe_interruption_repair_20260927.md)。

当前计划使用 `.sdsc/supervision/probe54485969-auto-v1/`，原观察器是
`.sdsc/supervision/probe54485969-readonly-v1/`。以当前 handoff 和目录里的
实际 `launch.json`、`state.json` 为准，目录存在本身不代表进程运行。

```bash
cd /gpfs/projects/p32737/del6500_home/OPD
cat .sdsc/supervision/probe54485969-auto-v1/state.json
cat .sdsc/supervision/probe54485969-auto-v1/notice.md
```

`notice.md` 在出现停止/完成等有意义的变化后生成。成功触发后还会保留
`continuation-last-message.txt`、JSONL 执行日志以及后继任务写入的
`continuation-result.json`。这些文件没有自动推送通知保证。新会话先读
这些记录以及 `docs/refactor/current_handoff.md`，不要再次启动同一 flow。

本次计划创建时使用以下入口；已启动后不要重跑：

```bash
tools/sdsc_auto_continue prepare \
  --job-id 54485969 \
  --watch-plan /absolute/path/to/probe54485969-readonly-v1/plan.json \
  --flow probe54485969-auto-v1 \
  --codex-bin /absolute/path/to/verified/codex \
  --authorize
tools/sdsc_auto_continue run --plan /absolute/path/to/probe54485969-auto-v1/plan.json
```

当前入口和固定接续任务只支持 v7 作业 54485969，不能直接拿这个任务文档
为不同科学版本授权；新版本需要自己的具体审查、任务和新计划。不要复制旧状态
文件、删 claim 或通过换目录重复触发。

## SSH 与离线边界

不需要人工盯着训练日志，但 Quest 主机、后继进程、Codex 账户连接及 SDSC
共享 SSH 必须可用。2026-09-27 的实测 master 是前台 SSH，关联 `pts/105`；
关闭该认证终端可能断开连接。当前流程不会改变用户的 SSH 进程。

下次需要手动认证时，可在同一 Quest 主机由用户使用后台 master：

```bash
mkdir -p "$HOME/.ssh/cm"
chmod 700 "$HOME/.ssh/cm"
ssh -MNf -o ControlMaster=yes -o ControlPersist=12h \
  -o ControlPath="$HOME/.ssh/cm/sdsc-$(hostname -s)" \
  zgao12@login.expanse.sdsc.edu
```

所有项目脚本始终使用这个运行时计算的明确路径。不要在现有 flow 活跃时退出共享 master 来
测试认证；SSH 丢失后先手动恢复，再核对停止状态及未知回执。
